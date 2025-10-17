/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
#ifndef APISessionState_h
#define APISessionState_h

#include "APIObject.h"
#include "SessionState.h"

namespace API {

class SessionState final : public ObjectImpl<Object::Type::SessionState> {
public:
    static Ref<SessionState> create(WebKit::SessionState);
    virtual ~SessionState();

    const WebKit::SessionState& sessionState() const { return m_sessionState; }

private:
    explicit SessionState(WebKit::SessionState);

    const WebKit::SessionState m_sessionState;
};

} // namespace API

#endif // APISessionState_h
