/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#ifdef __cplusplus
#include <wtf/text/WTFString.h>

extern "C" {

typedef void (^WGPUBufferMapBlockCallback)(WGPUBufferMapAsyncStatus);
typedef void (^WGPUCompilationInfoBlockCallback)(WGPUCompilationInfoRequestStatus, const WGPUCompilationInfo* compilationInfo);
typedef void (^WGPUCreateComputePipelineAsyncBlockCallback)(WGPUCreatePipelineAsyncStatus, WGPUComputePipeline pipeline, WTF::String&& message);
typedef void (^WGPUCreateRenderPipelineAsyncBlockCallback)(WGPUCreatePipelineAsyncStatus, WGPURenderPipeline pipeline, WTF::String&& message);
typedef void (^WGPUErrorBlockCallback)(WGPUErrorType, const char* message);
typedef void (^WGPUQueueWorkDoneBlockCallback)(WGPUQueueWorkDoneStatus);
typedef void (^WGPURequestAdapterBlockCallback)(WGPURequestAdapterStatus, WGPUAdapter adapter, const char* message);
typedef void (^WGPURequestDeviceBlockCallback)(WGPURequestDeviceStatus, WGPUDevice device, const char* message);
typedef void (^WGPURequestInvalidDeviceBlockCallback)(WGPUDevice);

#if !defined(WGPU_SKIP_PROCS)

typedef void (*WGPUProcAdapterRequestDeviceWithBlock)(WGPUAdapter, const WGPUDeviceDescriptor*, WGPURequestDeviceBlockCallback);
typedef void (*WGPUProcBufferMapAsyncWithBlock)(WGPUBuffer, WGPUMapModeFlags mode, size_t offset, size_t size, WGPUBufferMapBlockCallback);
typedef void (*WGPUProcDeviceCreateComputePipelineAsyncWithBlock)(WGPUDevice, const WGPUComputePipelineDescriptor* descriptor, WGPUCreateComputePipelineAsyncBlockCallback);
typedef void (*WGPUProcDeviceCreateRenderPipelineAsyncWithBlock)(WGPUDevice, const WGPURenderPipelineDescriptor* descriptor, WGPUCreateRenderPipelineAsyncBlockCallback);
typedef bool (*WGPUProcDevicePopErrorScopeWithBlock)(WGPUDevice, WGPUErrorBlockCallback);
typedef void (*WGPUProcDeviceSetUncapturedErrorCallbackWithBlock)(WGPUDevice, WGPUErrorBlockCallback);
typedef void (*WGPUProcInstanceRequestAdapterWithBlock)(WGPUInstance, const WGPURequestAdapterOptions*, WGPURequestAdapterBlockCallback);
typedef void (*WGPUProcQueueOnSubmittedWorkDoneWithBlock)(WGPUQueue, WGPUQueueWorkDoneBlockCallback);
typedef void (*WGPUProcShaderModuleGetCompilationInfoWithBlock)(WGPUShaderModule, WGPUCompilationInfoBlockCallback);

#endif // !defined(WGPU_SKIP_PROCS)

#if !defined(WGPU_SKIP_DECLARATIONS)

void wgpuAdapterRequestDeviceWithBlock(WGPUAdapter, const WGPUDeviceDescriptor*, WGPURequestDeviceBlockCallback);
void wgpuBufferMapAsyncWithBlock(WGPUBuffer, WGPUMapModeFlags, size_t offset, size_t, WGPUBufferMapBlockCallback);
void wgpuDeviceCreateComputePipelineAsyncWithBlock(WGPUDevice, const WGPUComputePipelineDescriptor*, WGPUCreateComputePipelineAsyncBlockCallback);
void wgpuDeviceCreateRenderPipelineAsyncWithBlock(WGPUDevice, const WGPURenderPipelineDescriptor*, WGPUCreateRenderPipelineAsyncBlockCallback);
void wgpuDevicePopErrorScopeWithBlock(WGPUDevice, WGPUErrorBlockCallback);
void wgpuDeviceSetUncapturedErrorCallbackWithBlock(WGPUDevice, WGPUErrorBlockCallback);
void wgpuInstanceRequestAdapterWithBlock(WGPUInstance, const WGPURequestAdapterOptions*, WGPURequestAdapterBlockCallback);
void wgpuQueueOnSubmittedWorkDoneWithBlock(WGPUQueue, WGPUQueueWorkDoneBlockCallback);
void wgpuShaderModuleGetCompilationInfoWithBlock(WGPUShaderModule, WGPUCompilationInfoBlockCallback);

#endif // !defined(WGPU_SKIP_DECLARATIONS)

} // extern "C"
#endif
