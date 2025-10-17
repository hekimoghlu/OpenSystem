/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#ifdef USE_TCL_STUBS

#include <dom.h>
#include <tdom.h>

/* !BEGIN!: Do not edit below this line. */

TdomStubs tdomStubs = {
    TCL_STUB_MAGIC,
    NULL,
    TclExpatObjCmd, /* 0 */
    CheckExpatParserObj, /* 1 */
    CHandlerSetInstall, /* 2 */
    CHandlerSetRemove, /* 3 */
    CHandlerSetCreate, /* 4 */
    CHandlerSetGet, /* 5 */
    CHandlerSetGetUserData, /* 6 */
    GetExpatInfo, /* 7 */
    XML_GetCurrentLineNumber, /* 8 */
    XML_GetCurrentColumnNumber, /* 9 */
    XML_GetCurrentByteIndex, /* 10 */
    XML_GetCurrentByteCount, /* 11 */
    XML_SetBase, /* 12 */
    XML_GetBase, /* 13 */
    XML_GetSpecifiedAttributeCount, /* 14 */
    XML_GetIdAttributeIndex, /* 15 */
    tcldom_getNodeFromName, /* 16 */
    tcldom_getDocumentFromName, /* 17 */
};

/* !END!: Do not edit above this line. */

#endif /* USE_TCL_STUBS */

