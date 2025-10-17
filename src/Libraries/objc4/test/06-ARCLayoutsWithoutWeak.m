/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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

// Same as test ARCLayouts but with MRC __weak support disabled.
/*
TEST_CONFIG MEM=arc OS=!exclavekit
TEST_BUILD
    mkdir -p $T{OBJDIR}
    $C{COMPILE_NOLINK_NOMEM} -c $DIR/MRCBase.m -o $T{OBJDIR}/MRCBase.o -fno-objc-weak
    $C{COMPILE_NOLINK_NOMEM} -c $DIR/MRCARC.m  -o $T{OBJDIR}/MRCARC.o  -fno-objc-weak
    $C{COMPILE_NOLINK}       -c $DIR/ARCBase.m -o $T{OBJDIR}/ARCBase.o
    $C{COMPILE_NOLINK}       -c $DIR/ARCMRC.m  -o $T{OBJDIR}/ARCMRC.o
    $C{COMPILE} '-DNAME=\"06-ARCLayoutsWithoutWeak.m\"' -fobjc-arc $DIR/ARCLayouts.m -x none $T{OBJDIR}/MRCBase.o $T{OBJDIR}/MRCARC.o $T{OBJDIR}/ARCBase.o $T{OBJDIR}/ARCMRC.o -framework Foundation -o 06-ARCLayoutsWithoutWeak.exe
END
*/
