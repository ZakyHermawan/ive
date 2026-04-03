#include "IveToSCF.hpp"

#include "ive/Dialect.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace ive {

#define GEN_PASS_DEF_IVETOSCF
#include "IveToSCF.hpp.inc"

class IveToSCFTypeConverter : public TypeConverter {
public:
  IveToSCFTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
  }
};

struct ConvertYield : public OpRewritePattern<YieldOp> {
  ConvertYield(MLIRContext *context) : OpRewritePattern<YieldOp>(context) {}

  LogicalResult matchAndRewrite(YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp);
    return success();
  }
};

struct ConvertIf : public OpConversionPattern<IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp ifOp, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = ifOp.getLoc();
    Value conditionTensor = opAdaptor.getCondition();

    Value element =
        tensor::ExtractOp::create(rewriter, loc, conditionTensor, ValueRange{});
    Value zero = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF64Type(), llvm::APFloat(0.0));

    auto cmp = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::UNE,
                                     element, zero);
    Value condition = cmp.getResult();

    bool hasElseRegion = !ifOp.getElseRegion().empty();
    auto scfIf = scf::IfOp::create(rewriter, loc, condition, hasElseRegion);

    Block *srcThen = &ifOp.getThenRegion().front();
    Block *dstThen = &scfIf.getThenRegion().front();
    rewriter.eraseOp(dstThen->getTerminator());
    rewriter.inlineBlockBefore(srcThen, dstThen, dstThen->end());

    if (hasElseRegion) {
      Block *srcElse = &ifOp.getElseRegion().front();
      Block *dstElse = &scfIf.getElseRegion().front();
      rewriter.eraseOp(dstElse->getTerminator());
      rewriter.inlineBlockBefore(srcElse, dstElse, dstElse->end());
    }

    rewriter.eraseOp(ifOp);
    return success();
  }
};

struct ConvertFor : public OpConversionPattern<ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ForOp forOp, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (forOp.getPredicate() != "lt") {
      return rewriter.notifyMatchFailure(
          forOp, "only predicate 'lt' is currently supported");
    }

    Location loc = forOp.getLoc();

    auto toIndex = [&](Value tensorValue) -> Value {
      Value scalar =
          tensor::ExtractOp::create(rewriter, loc, tensorValue, ValueRange{});
      Value i64Val =
          arith::FPToSIOp::create(rewriter, loc, rewriter.getI64Type(), scalar);
      return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                        i64Val);
    };

    Value lowerBound = toIndex(opAdaptor.getLowerBound());
    Value upperBound = toIndex(opAdaptor.getUpperBound());
    Value step = toIndex(opAdaptor.getStep());

    auto scfFor =
        scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
    Block *dstBody = scfFor.getBody();

    rewriter.setInsertionPointToStart(dstBody);
    Value ivI64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                             scfFor.getInductionVar());
    Value ivF64 =
        arith::SIToFPOp::create(rewriter, loc, rewriter.getF64Type(), ivI64);
    auto scalarTensorTy = RankedTensorType::get({}, rewriter.getF64Type());
    Value ivTensor =
      tensor::SplatOp::create(rewriter, loc, scalarTensorTy, ivF64);

    rewriter.eraseOp(dstBody->getTerminator());
    Block *srcBody = &forOp.getBody().front();
    rewriter.inlineBlockBefore(srcBody, dstBody, dstBody->end(),
                               ValueRange{ivTensor});

    rewriter.eraseOp(forOp);
    return success();
  }
};

struct IveToSCF : impl::IveToSCFBase<IveToSCF> {
  using IveToSCFBase::IveToSCFBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ConversionTarget target(*context);
    // Convert ive.if/ive.for/ive.yield and keep all other ops legal.
    target.addIllegalOp<IfOp, ForOp, YieldOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(context);
    IveToSCFTypeConverter typeConverter(context);
    patterns.add<ConvertIf>(typeConverter, context);
    patterns.add<ConvertFor>(typeConverter, context);
    patterns.add<ConvertYield>(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace ive
} // namespace mlir
