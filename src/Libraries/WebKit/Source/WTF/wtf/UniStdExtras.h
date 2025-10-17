/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
#ifndef UniStdExtras_h
#define UniStdExtras_h

#include <errno.h>
#include <unistd.h>

namespace WTF {

WTF_EXPORT_PRIVATE bool setCloseOnExec(int fileDescriptor);
WTF_EXPORT_PRIVATE bool unsetCloseOnExec(int fileDescriptor);
WTF_EXPORT_PRIVATE int dupCloseOnExec(int fileDescriptor);

inline int closeWithRetry(int fileDescriptor)
{
    int ret;
#if OS(LINUX)
    // Workaround for the Linux behavior of closing the descriptor
    // unconditionally, even if the close() call is interrupted.
    // See https://bugs.webkit.org/show_bug.cgi?id=117266 for more
    // details.
    if ((ret = close(fileDescriptor)) == -1 && errno == EINTR)
        return 0;
#else
    while ((ret = close(fileDescriptor)) == -1 && errno == EINTR) { }
#endif
    return ret;
}

WTF_EXPORT_PRIVATE bool setNonBlock(int fileDescriptor);

} // namespace WTF

using WTF::closeWithRetry;
using WTF::dupCloseOnExec;
using WTF::setCloseOnExec;
using WTF::unsetCloseOnExec;
using WTF::setNonBlock;

#endif // UniStdExtras_h
