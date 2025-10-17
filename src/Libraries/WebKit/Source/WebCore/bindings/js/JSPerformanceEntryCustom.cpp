/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
#include "config.h"

#include "JSPerformanceEntry.h"

#include "JSDOMBinding.h"
#include "JSPerformanceMark.h"
#include "JSPerformanceMeasure.h"
#include "JSPerformanceNavigationTiming.h"
#include "JSPerformancePaintTiming.h"
#include "JSPerformanceResourceTiming.h"
#include "PerformanceMark.h"
#include "PerformanceMeasure.h"
#include "PerformanceNavigationTiming.h"
#include "PerformancePaintTiming.h"
#include "PerformanceResourceTiming.h"


namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<PerformanceEntry>&& entry)
{
    switch (entry->performanceEntryType()) {
    case PerformanceEntry::Type::Navigation:
        return createWrapper<PerformanceNavigationTiming>(globalObject, WTFMove(entry));
    case PerformanceEntry::Type::Mark:
        return createWrapper<PerformanceMark>(globalObject, WTFMove(entry));
    case PerformanceEntry::Type::Measure:
        return createWrapper<PerformanceMeasure>(globalObject, WTFMove(entry));
    case PerformanceEntry::Type::Resource:
        return createWrapper<PerformanceResourceTiming>(globalObject, WTFMove(entry));
    case PerformanceEntry::Type::Paint:
        return createWrapper<PerformancePaintTiming>(globalObject, WTFMove(entry));
    }

    ASSERT_NOT_REACHED();
    return createWrapper<PerformanceEntry>(globalObject, WTFMove(entry));
}

JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, PerformanceEntry& entry)
{
    return wrap(lexicalGlobalObject, globalObject, entry);
}

} // namespace WebCore
