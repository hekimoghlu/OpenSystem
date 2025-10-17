/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#if !defined(__JSC_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <jsc/jsc.h> can be included directly."
#endif

#ifndef JSCAutoPtr_h
#define JSCAutoPtr_h

#ifdef G_DEFINE_AUTOPTR_CLEANUP_FUNC
#ifndef __GI_SCANNER__

G_DEFINE_AUTOPTR_CLEANUP_FUNC (JSCClass, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (JSCContext, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (JSCException, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (JSCValue, g_object_unref)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (JSCVirtualMachine, g_object_unref)

#endif /* __GI_SCANNER__ */
#endif /* G_DEFINE_AUTOPTR_CLEANUP_FUNC */

#endif /* JSCAutoPtr_h */
