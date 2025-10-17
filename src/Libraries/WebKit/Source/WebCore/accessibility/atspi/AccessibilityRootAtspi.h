/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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

#if USE(ATSPI)
#include "AccessibilityAtspiEnums.h"
#include "IntRect.h"
#include <wtf/FastMalloc.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

typedef struct _GDBusInterfaceVTable GDBusInterfaceVTable;
typedef struct _GVariant GVariant;

namespace WebCore {
class AccessibilityObjectAtspi;
class Page;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(AccessibilityRootAtspi);
class AccessibilityRootAtspi final : public RefCountedAndCanMakeWeakPtr<AccessibilityRootAtspi> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(AccessibilityRootAtspi);
public:
    static Ref<AccessibilityRootAtspi> create(Page&);
    ~AccessibilityRootAtspi() = default;

    void registerObject(CompletionHandler<void(const String&)>&&);
    void unregisterObject();
    void setPath(String&&);

    const String& path() const { return m_path; }
    GVariant* reference() const;
    GVariant* parentReference() const;
    GVariant* applicationReference() const;
    AccessibilityObjectAtspi* child() const;
    void childAdded(AccessibilityObjectAtspi&);
    void childRemoved(AccessibilityObjectAtspi&);

    void serialize(GVariantBuilder*) const;

private:
    explicit AccessibilityRootAtspi(Page&);

    void embedded(const char* parentUniqueName, const char* parentPath);
    IntRect frameRect(Atspi::CoordinateType) const;

    static GDBusInterfaceVTable s_accessibleFunctions;
    static GDBusInterfaceVTable s_socketFunctions;
    static GDBusInterfaceVTable s_componentFunctions;

    WeakPtr<Page> m_page;
    String m_path;
    String m_parentUniqueName;
    String m_parentPath;
};

} // namespace WebCore

#endif // USE(ATSPI)
