/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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

#include <memory>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class HTMLInputElement;
class RadioButtonGroup;

class RadioButtonGroups {
    WTF_MAKE_TZONE_ALLOCATED(RadioButtonGroups);
public:
    RadioButtonGroups();
    ~RadioButtonGroups();
    void addButton(HTMLInputElement&);
    void updateCheckedState(HTMLInputElement&);
    void requiredStateChanged(HTMLInputElement&);
    void removeButton(HTMLInputElement&);
    RefPtr<HTMLInputElement> checkedButtonForGroup(const AtomString& groupName) const;
    bool hasCheckedButton(const HTMLInputElement&) const;
    bool isInRequiredGroup(HTMLInputElement&) const;
    Vector<Ref<HTMLInputElement>> groupMembers(const HTMLInputElement&) const;

private:
    typedef UncheckedKeyHashMap<AtomString, std::unique_ptr<RadioButtonGroup>> NameToGroupMap;
    NameToGroupMap m_nameToGroupMap;
};

} // namespace WebCore
