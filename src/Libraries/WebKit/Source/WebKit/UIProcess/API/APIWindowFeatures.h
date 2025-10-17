/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#ifndef APIWindowFeatures_h
#define APIWindowFeatures_h

#include "APIObject.h"
#include <WebCore/WindowFeatures.h>

namespace API {

class WindowFeatures final : public ObjectImpl<Object::Type::WindowFeatures> {
public:
    static Ref<WindowFeatures> create(const WebCore::WindowFeatures&);
    virtual ~WindowFeatures();

    const WebCore::WindowFeatures& windowFeatures() const { return m_windowFeatures; }

private:
    explicit WindowFeatures(const WebCore::WindowFeatures&);

    const WebCore::WindowFeatures m_windowFeatures;
};

}

#endif // APIWindowFeatures_h
