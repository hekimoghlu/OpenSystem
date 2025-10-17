/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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

#include <wtf/text/WTFString.h>

namespace WTF {

WTF_EXPORT_PRIVATE bool isInsideFlatpak();
WTF_EXPORT_PRIVATE bool isInsideUnsupportedContainer();
WTF_EXPORT_PRIVATE bool isInsideSnap();
WTF_EXPORT_PRIVATE bool shouldUseBubblewrap();
WTF_EXPORT_PRIVATE bool shouldUsePortal();
WTF_EXPORT_PRIVATE bool checkFlatpakPortalVersion(int);

WTF_EXPORT_PRIVATE const CString& sandboxedUserRuntimeDirectory();

} // namespace WTF

using WTF::isInsideFlatpak;
using WTF::isInsideUnsupportedContainer;
using WTF::isInsideSnap;
using WTF::shouldUseBubblewrap;
using WTF::shouldUsePortal;
using WTF::checkFlatpakPortalVersion;

using WTF::sandboxedUserRuntimeDirectory;
