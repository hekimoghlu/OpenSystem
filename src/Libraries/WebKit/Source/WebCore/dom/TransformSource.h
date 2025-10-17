/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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

#if ENABLE(XSLT)

// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
IGNORE_WARNINGS_BEGIN("undef")
#include <libxml/tree.h>
IGNORE_WARNINGS_END
IGNORE_WARNINGS_END
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

typedef xmlDocPtr PlatformTransformSource;

class TransformSource {
    WTF_MAKE_TZONE_ALLOCATED(TransformSource);
    WTF_MAKE_NONCOPYABLE(TransformSource);
public:
    explicit TransformSource(const PlatformTransformSource&);
    ~TransformSource();

    PlatformTransformSource platformSource() const { return m_source; }

private:
    PlatformTransformSource m_source;
};

} // namespace WebCore

#endif // ENABLE(XSLT)
