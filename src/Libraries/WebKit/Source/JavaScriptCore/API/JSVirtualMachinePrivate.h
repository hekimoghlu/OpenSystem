/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#import "JSExportMacros.h"
#import <JavaScriptCore/JavaScript.h>

#if JSC_OBJC_API_ENABLED

#import <JavaScriptCore/JSVirtualMachine.h>

@interface JSVirtualMachine(JSPrivate)

/*!
@method
@discussion Shrinks the memory footprint of the VM by deleting various internal caches,
 running synchronous garbage collection, and releasing memory back to the OS. Note: this
 API waits until no JavaScript is running on the stack before it frees any memory. It's
 best to call this API when no JavaScript is running on the stack for this reason. However, if
 you do call this API when JavaScript is running on the stack, the API will wait until all JavaScript
 on the stack finishes running to free memory back to the OS. Therefore, calling this
 API may not synchronously free memory.
*/

- (void)shrinkFootprintWhenIdle JSC_API_AVAILABLE(macos(10.14), ios(12.0));

/*!
@method
@abstract Set the number of threads to be used by the DFG JIT compiler.
@discussion If called after the VM has been initialized, it will terminate
 threads until it meets the new limit or create new threads accordingly if the
 new limit is higher than the previous limit. If called before initialization,
 the Options value for the number of DFG threads will be updated to ensure the
 DFG compiler already starts with the up-to-date limit.
@param numberOfThreads The number of threads the DFG compiler should use going forward
@result The previous number of threads being used by the DFG compiler
*/
+ (NSUInteger)setNumberOfDFGCompilerThreads:(NSUInteger)numberOfThreads JSC_API_AVAILABLE(macos(10.14), ios(12.0));

/*!
@method
@abstract Set the number of threads to be used by the FTL JIT compiler.
@discussion If called after the VM has been initialized, it will terminate
 threads until it meets the new limit or create new threads accordingly if the
 new limit is higher than the previous limit. If called before initialization,
 the Options value for the number of FTL threads will be updated to ensure the
 FTL compiler already starts with the up-to-date limit.
@param numberOfThreads The number of threads the FTL compiler should use going forward
@result The previous number of threads being used by the FTL compiler
*/
+ (NSUInteger)setNumberOfFTLCompilerThreads:(NSUInteger)numberOfThreads JSC_API_AVAILABLE(macos(10.14), ios(12.0));

/*!
@method
@abstract Allows embedders of JSC to specify that JSC should crash the process if a VM is created when unexpected.
@param shouldCrash Sets process-wide state that indicates whether VM creation should crash or not.
*/
+ (void)setCrashOnVMCreation:(BOOL)shouldCrash;

@end

#endif // JSC_OBJC_API_ENABLED
