/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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

// This file's only purpose is to collect commonly used *Inlines.h files, so that you don't
// have to include all of them in every .cpp file. Instead you just include this. It's good
// style to make sure that every .cpp file includes JSCInlines.h.
//
// JSC should never include this file, or any *Inline.h file, from interface headers, since
// this could lead to a circular dependency.
//
// WebCore, or any other downstream client of JSC, is allowed to include this file in headers.
// In fact, it can make a lot of sense: outside of JSC, this file becomes a kind of umbrella
// header that pulls in most (all?) of the interesting things in JSC.

#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#include "ExceptionHelpers.h"
#include "GCIncomingRefCountedInlines.h"
#include "HeapInlines.h"
#include "IdentifierInlines.h"
#include "JSArrayBufferViewInlines.h"
#include "JSCJSValueInlines.h"
#include "JSCellInlines.h"
#include "JSFunctionInlines.h"
#include "JSGlobalObjectInlines.h"
#include "JSGlobalProxy.h"
#include "JSObjectInlines.h"
#include "JSString.h"
#include "Operations.h"
#include "SlotVisitorInlines.h"
#include "StrongInlines.h"
#include "StructureInlines.h"
#include "ThrowScope.h"
#include "WeakGCMapInlines.h"
#include "WeakGCSetInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
