/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

#include <wtf/RetainPtr.h>

#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
namespace IPC {

template<typename T>
class CoreIPCRetainPtr : public RetainPtr<T> {
public:
    CoreIPCRetainPtr() = default;
    CoreIPCRetainPtr(T *object)
        : RetainPtr<T>(object) { }
    CoreIPCRetainPtr(RetainPtr<T>&& object)
        : RetainPtr<T>(WTFMove(object)) { }
};

}
#endif // !HAVE(WK_SECURE_CODING_NSURLREQUEST)
