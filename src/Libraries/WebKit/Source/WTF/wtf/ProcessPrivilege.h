/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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

#include <wtf/OptionSet.h>

namespace WTF {

enum class ProcessPrivilege {
    CanAccessRawCookies            = 1 << 0,
    CanAccessCredentials           = 1 << 1,
    CanCommunicateWithWindowServer = 1 << 2,
};

WTF_EXPORT_PRIVATE void setProcessPrivileges(OptionSet<ProcessPrivilege>);
WTF_EXPORT_PRIVATE void addProcessPrivilege(ProcessPrivilege);
WTF_EXPORT_PRIVATE void removeProcessPrivilege(ProcessPrivilege);
WTF_EXPORT_PRIVATE bool hasProcessPrivilege(ProcessPrivilege);
WTF_EXPORT_PRIVATE OptionSet<ProcessPrivilege> allPrivileges();

} // namespace WTF

using WTF::ProcessPrivilege;
using WTF::allPrivileges;
using WTF::hasProcessPrivilege;
using WTF::setProcessPrivileges;

