/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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

#include "APIObject.h"
#include <wtf/RefPtr.h>

namespace WebKit {

class UserData {
public:
    UserData();
    explicit UserData(RefPtr<API::Object>&&);
    ~UserData();

    struct Transformer {
        virtual ~Transformer() { }
        virtual bool shouldTransformObject(const API::Object&) const = 0;
        virtual RefPtr<API::Object> transformObject(API::Object&) const = 0;
    };
    static RefPtr<API::Object> transform(API::Object*, const Transformer&);

    API::Object* object() const { return m_object.get(); }
    RefPtr<API::Object> protectedObject() const { return m_object; }

private:
    RefPtr<API::Object> m_object;
};

} // namespace WebKit
