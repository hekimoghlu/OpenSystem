/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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

#include "PluginData.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMPlugin;
class Navigator;

class DOMMimeType : public RefCounted<DOMMimeType> {
public:
    static Ref<DOMMimeType> create(Navigator&, const MimeClassInfo&, DOMPlugin&);
    ~DOMMimeType();

    AtomString type() const;
    String suffixes() const;
    String description() const;
    RefPtr<DOMPlugin> enabledPlugin() const;

    Navigator* navigator() { return m_navigator.get(); }

private:
    DOMMimeType(Navigator&, const MimeClassInfo&, DOMPlugin&);

    WeakPtr<Navigator> m_navigator;
    MimeClassInfo m_info;
    WeakPtr<DOMPlugin> m_enabledPlugin;
};

} // namespace WebCore
