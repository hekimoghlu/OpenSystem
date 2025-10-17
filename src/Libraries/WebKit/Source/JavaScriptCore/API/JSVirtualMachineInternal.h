/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
#ifndef JSVirtualMachineInternal_h
#define JSVirtualMachineInternal_h

#if JSC_OBJC_API_ENABLED

#import <JavaScriptCore/JavaScriptCore.h>

namespace JSC {
class VM;
class AbstractSlotVisitor;
}

#if defined(__OBJC__)
@class NSMapTable;

@interface JSVirtualMachine(Internal)

JSContextGroupRef getGroupFromVirtualMachine(JSVirtualMachine *);

+ (JSVirtualMachine *)virtualMachineWithContextGroupRef:(JSContextGroupRef)group;

- (JSContext *)contextForGlobalContextRef:(JSGlobalContextRef)globalContext;
- (void)addContext:(JSContext *)wrapper forGlobalContextRef:(JSGlobalContextRef)globalContext;
- (BOOL)isWebThreadAware;

@property (readonly) JSContextGroupRef JSContextGroupRef;

@end

#endif // defined(__OBJC__)

void scanExternalObjectGraph(JSC::VM&, JSC::AbstractSlotVisitor&, void* root);
void scanExternalRememberedSet(JSC::VM&, JSC::AbstractSlotVisitor&);

#endif // JSC_OBJC_API_ENABLED

#endif // JSVirtualMachineInternal_h
