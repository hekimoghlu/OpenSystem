/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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

#include "JSDOMConvertDictionary.h"
#include "TestDerivedDictionary2.h"

namespace WebCore {

template<> ConversionResult<IDLDictionary<TestDerivedDictionary2>> convertDictionary<TestDerivedDictionary2>(JSC::JSGlobalObject&, JSC::JSValue);

JSC::JSObject* convertDictionaryToJS(JSC::JSGlobalObject&, JSDOMGlobalObject&, const TestDerivedDictionary2&);

template<> ConversionResult<IDLDictionary<TestDerivedDictionary2::Dictionary>> convertDictionary<TestDerivedDictionary2::Dictionary>(JSC::JSGlobalObject&, JSC::JSValue);

JSC::JSObject* convertDictionaryToJS(JSC::JSGlobalObject&, JSDOMGlobalObject&, const TestDerivedDictionary2::Dictionary&);

} // namespace WebCore
