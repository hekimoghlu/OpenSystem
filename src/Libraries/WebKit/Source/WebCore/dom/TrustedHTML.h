/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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

#include "ScriptWrappable.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class WEBCORE_EXPORT TrustedHTML : public ScriptWrappable, public RefCounted<TrustedHTML> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(TrustedHTML, WEBCORE_EXPORT);
public:
    static Ref<TrustedHTML> create(const String& data);
    ~TrustedHTML() = default;

    String toString() const { return m_data; }
    String toJSON() const { return toString(); }

private:
    TrustedHTML(const String& data);

    const String m_data;
};

} // namespace WebCore
