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
// STL
#include <vector>
#include <span>

// mach_o
#include "Header.h"
#include "Error.h"

// common
#include "Defines.h"
#include "CString.h"


struct VerifierError
{
    VerifierError(CString name) : verifierErrorName(name) { }

    CString         verifierErrorName;
    mach_o::Error   message;
};


/*!
 * @function os_macho_verifier
 *
 * @abstract
 *      Used by B&I verifer to ensure binaries follow Apple's rules for OS mach-o files
 *
 * @param path
 *      Full path to file (in $DSTROOT) to examine.
 *
 * @param buffer
 *      The content of the file.
 *
 * @param verifierDstRoot
 *      $DSTROOT path.
 *
 * @param mergeRootPaths
 *      If B&I moves content file system location
 *
 * @param errors
 *      For each error found in file, a VerifierError is added to this vector.
 *
 */
void os_macho_verifier(CString path, std::span<const uint8_t> buffer, CString verifierDstRoot,
                       const std::vector<CString>& mergeRootPaths, std::vector<VerifierError>& errors);





