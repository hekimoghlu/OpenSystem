/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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

#include "AXCoreObject.h"
#include "AXObjectCache.h"
#include <wtf/MonotonicTime.h>

namespace WebCore {

enum class AXLoggingOptions : uint8_t {
    MainThread = 1 << 0, // Logs messages on the main thread.
    OffMainThread = 1 << 1, // Logs messages off the main thread.
};

enum class AXStreamOptions : uint8_t {
    ObjectID = 1 << 0,
    Role = 1 << 1,
    ParentID = 1 << 2,
    IdentifierAttribute = 1 << 3,
    OuterHTML = 1 << 4,
    DisplayContents = 1 << 5,
    Address = 1 << 6,
#if ENABLE(AX_THREAD_TEXT_APIS)
    TextRuns = 1 << 7,
#endif
};

#if !LOG_DISABLED

class AXLogger final {
public:
    AXLogger() = default;
    AXLogger(const String& methodName);
    ~AXLogger();
    void log(const String&);
    void log(const char*);
    void log(const AXCoreObject&);
    void log(RefPtr<AXCoreObject>);
    void log(const Vector<Ref<AXCoreObject>>&);
    void log(const std::pair<Ref<AccessibilityObject>, AXNotification>&);
    void log(const std::pair<RefPtr<AXCoreObject>, AXNotification>&);
    void log(const AccessibilitySearchCriteria&);
    void log(AccessibilityObjectInclusion);
    void log(AXRelationType);
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    void log(AXIsolatedTree&);
#endif
    void log(AXObjectCache&);
    static void add(TextStream&, const RefPtr<AXCoreObject>&, bool recursive = false);
    void log(const String&, const AXObjectCache::DeferredCollection&);
private:
    bool shouldLog();
    String m_methodName;
    MonotonicTime m_startTime;
};

#define AXTRACE(methodName) AXLogger axLogger(methodName)
#define AXLOG(x) axLogger.log(x)
#define AXLOGDeferredCollection(name, collection) axLogger.log(name, collection)

#else

#define AXTRACE(methodName) (void)0
#define AXLOG(x) (void)0
#define AXLOGDeferredCollection(name, collection) (void)0

#endif // !LOG_DISABLED

void streamAXCoreObject(TextStream&, const AXCoreObject&, const OptionSet<AXStreamOptions>&);
void streamSubtree(TextStream&, const Ref<AXCoreObject>&, const OptionSet<AXStreamOptions>&);

} // namespace WebCore
