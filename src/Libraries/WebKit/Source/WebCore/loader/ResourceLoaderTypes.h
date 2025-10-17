/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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

namespace WebCore {

// DataPayloadType describes the nature of an invocation of the ResourceLoader::didReceiveData callback.
//  - DataPayloadWholeResource indicates that the buffer points to a whole resource. There will only be one such didReceiveData callback for the load.
//  - DataPayloadBytes indicates that the buffer points to a range of bytes, which may or may not be a whole resource.
//    There may have been previous didReceieveData callbacks, and there may be future didReceieveData callbacks.

enum DataPayloadType {
    DataPayloadWholeResource,
    DataPayloadBytes,
};

} // namespace WebCore
