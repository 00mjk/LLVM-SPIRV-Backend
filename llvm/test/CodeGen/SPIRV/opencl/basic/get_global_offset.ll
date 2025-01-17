; RUN: llc -O0 %s -o - | FileCheck %s

target triple = "spirv64-unknown-unknown"

; CHECK: OpEntryPoint Kernel %[[test_func:[0-9]+]] "test"
; CHECK: OpName %[[outOffsets:[0-9]+]] "outOffsets"
; CHECK: OpName %[[test_func]] "test"
; CHECK: OpName %[[f2_decl:[0-9]+]] "BuiltInGlobalOffset"
; CHECK: OpDecorate %[[f2_decl]] LinkageAttributes "BuiltInGlobalOffset" Import
; CHECK: %[[int_ty:[0-9]+]] = OpTypeInt 32 0
; CHECK: %[[iptr_ty:[0-9]+]] = OpTypePointer CrossWorkgroup  %[[int_ty]]
; CHECK: %[[void_ty:[0-9]+]] = OpTypeVoid
; CHECK: %[[func_ty:[0-9]+]] = OpTypeFunction %[[void_ty]] %[[iptr_ty]]
; CHECK: %[[int64_ty:[0-9]+]] = OpTypeInt 64 0
; CHECK: %[[vec_ty:[0-9]+]] = OpTypeVector %[[int64_ty]] 3
; CHECK: %[[func2_ty:[0-9]+]] = OpTypeFunction %[[vec_ty]]
; TODO: add 64-bit constant defs
; CHECK: %[[f2_decl]] = OpFunction %[[vec_ty]] Pure %[[func2_ty]]
; CHECK: OpFunctionEnd
; Check that the function register name does not match other registers
; CHECK-NOT: %[[int_ty]] = OpFunction
; CHECK-NOT: %[[iptr_ty]] = OpFunction
; CHECK-NOT: %[[void_ty]] = OpFunction
; CHECK-NOT: %[[func_ty]] = OpFunction
; CHECK-NOT: %[[int64_ty]] = OpFunction
; CHECK-NOT: %[[vec_ty]] = OpFunction
; CHECK-NOT: %[[func2_ty]] = OpFunction
; CHECK-NOT: %[[f2_decl]] = OpFunction
; CHECK: %[[outOffsets]] = OpFunctionParameter %[[iptr_ty]]
; Function Attrs: nounwind
define spir_kernel void @test(i32 addrspace(1)* %outOffsets) #0 {
entry:
  %0 = call spir_func <3 x i64> @BuiltInGlobalOffset() #1
  %call = extractelement <3 x i64> %0, i32 0
  %conv = trunc i64 %call to i32
; CHECK: %[[i1:[0-9]+]] = OpInBoundsPtrAccessChain %[[iptr_ty]] %[[outOffsets]]
; CHECK: OpStore %[[i1:[0-9]+]] %[[#]] Aligned 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %outOffsets, i64 0
  store i32 %conv, i32 addrspace(1)* %arrayidx, align 4
  %1 = call spir_func <3 x i64> @BuiltInGlobalOffset() #1
  %call1 = extractelement <3 x i64> %1, i32 1
  %conv2 = trunc i64 %call1 to i32
; CHECK: %[[i2:[0-9]+]] = OpInBoundsPtrAccessChain %[[iptr_ty]] %[[outOffsets]]
; CHECK: OpStore %[[i2:[0-9]+]] %[[#]] Aligned 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %outOffsets, i64 1
  store i32 %conv2, i32 addrspace(1)* %arrayidx3, align 4
  %2 = call spir_func <3 x i64> @BuiltInGlobalOffset() #1
  %call4 = extractelement <3 x i64> %2, i32 2
  %conv5 = trunc i64 %call4 to i32
; CHECK: %[[i3:[0-9]+]] = OpInBoundsPtrAccessChain %[[iptr_ty]] %[[outOffsets]]
; CHECK: OpStore %[[i3:[0-9]+]] %[[#]] Aligned 4
  %arrayidx6 = getelementptr inbounds i32, i32 addrspace(1)* %outOffsets, i64 2
  store i32 %conv5, i32 addrspace(1)* %arrayidx6, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func <3 x i64> @BuiltInGlobalOffset() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.kernels = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!166}
!opencl.ocl.version = !{!166}
!opencl.used.extensions = !{!167}
!opencl.used.optional.core.features = !{!167}
!opencl.compiler.options = !{!168}

!0 = !{void (i32 addrspace(1)*)* @test, !1, !2, !3, !4, !5, !6}
!1 = !{!"kernel_arg_addr_space", i32 1}
!2 = !{!"kernel_arg_access_qual", !"none"}
!3 = !{!"kernel_arg_type", !"int*"}
!4 = !{!"kernel_arg_type_qual", !""}
!5 = !{!"kernel_arg_base_type", !"int*"}
!6 = !{!"kernel_arg_name", !"outOffsets"}
!166 = !{i32 3, i32 0}
!167 = !{}
!168 = !{!"-cl-std=CL3.0"}
