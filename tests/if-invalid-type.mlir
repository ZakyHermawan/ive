// RUN: not %ive %s -emit=mlir 2>&1 | FileCheck %s
// CHECK: 'ive.if' op operand #0 must be 1-bit signless integer
module {
  ive.func @main() {
    %count = ive.scalar_constant 1 : i32
    "ive.if"(%count) ({
      "ive.yield"() : () -> ()
    }, {}) : (i32) -> ()
    ive.return
  }
}
