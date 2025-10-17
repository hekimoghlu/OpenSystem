/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 7, 2023.
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

#include <wtf/Forward.h>

namespace WTF {

// This is strerror, except it is threadsafe. The problem with normal strerror is it returns a
// pointer to static storage, and it may actually modify that storage, so it can never be used in
// any multithreaded application, or any library that may be linked to a multithreaded application.
// (Why does it modify its storage? So that it can append the error number to the error string, as
// in "Unknown error n." Also, because it will localize the error message.) The standard
// alternatives are strerror_s and strerror_r, but both have problems. strerror_s is specified by
// C11, but not by C++ (as of C++20), and it is optional so glibc decided to ignore it. We can only
// rely on it to exist on Windows. Then strerror_r is even worse. First, it doesn't exist at all on
// Windows. Second, the GNU version is incompatible with the POSIX version, and it is impossible to
// use correctly unless you know which version you have. Both strerror_s and strerror_r are
// cumbersome because they force you to allocate the buffer for the result manually. It's all such a
// mess that we should deal with the complexity here rather than elsewhere in WebKit.
WTF_EXPORT_PRIVATE CString safeStrerror(int errnum);

}

using WTF::safeStrerror;
