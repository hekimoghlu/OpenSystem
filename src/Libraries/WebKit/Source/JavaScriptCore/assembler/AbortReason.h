/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

namespace JSC {

// It's important to not change the values of existing abort reasons unless we really
// have to. For this reason there is a BASIC-style numbering that should allow us to
// sneak new reasons in without changing the numbering of existing reasons - at least
// for a while.
enum AbortReason {
    AHCallFrameMisaligned                             =  10,
    AHIndexingTypeIsValid                             =  20,
    AHInsaneArgumentCount                             =  30,
    AHIsNotCell                                       =  40,
    AHIsNotInt32                                      =  50,
    AHIsNotJSDouble                                   =  60,
    AHIsNotJSInt32                                    =  70,
    AHIsNotJSNumber                                   =  80,
    AHIsNotNull                                       =  90,
    AHStackPointerMisaligned                          = 100,
    AHInvalidCodeBlock                                = 101,
    AHStructureIDIsValid                              = 110,
    AHNotCellMaskNotInPlace                           = 120,
    AHNumberTagNotInPlace                             = 130,
    AHTypeInfoInlineTypeFlagsAreValid                 = 140,
    AHTypeInfoIsValid                                 = 150,
    B3Oops                                            = 155,
    DFGBailedAtTopOfBlock                             = 161,
    DFGBailedAtEndOfNode                              = 162,
    DFGBasicStorageAllocatorZeroSize                  = 170,
    DFGIsNotCell                                      = 180,
    DFGIneffectiveWatchpoint                          = 190,
    DFGNegativeStringLength                           = 200,
    DFGSlowPathGeneratorFellThrough                   = 210,
    DFGUnreachableBasicBlock                          = 220,
    DFGUnreachableNode                                = 225,
    DFGUnreasonableOSREntryJumpDestination            = 230,
    DFGVarargsThrowingPathDidNotThrow                 = 235,
    FTLCrash                                          = 236,
    JITDidReturnFromTailCall                          = 237,
    JITDivOperandsAreNotNumbers                       = 240,
    JITGetByValResultIsNotEmpty                       = 250,
    JITNotSupported                                   = 260,
    JITOffsetIsNotOutOfLine                           = 270,
    JITUncaughtExceptionAfterCall                     = 275,
    JITUnexpectedCallFrameSize                        = 277,
    JITUnreasonableLoopHintJumpTarget                 = 280,
    MacroAssemblerOops                                = 285,
    RPWUnreasonableJumpTarget                         = 290,
    RepatchIneffectiveWatchpoint                      = 300,
    RepatchInsaneArgumentCount                        = 310,
    TGInvalidPointer                                  = 320,
    TGNotSupported                                    = 330,
    UncheckedOverflow                                 = 335,
    VMCreationDisallowed                              = 998,
    VMEntryDisallowed                                 = 999,
};

// This enum is for CRASH_WITH_SECURITY_IMPLICATION_AND_INFO so we can easily identify which assertion
// we are looking at even if the calls to crash get coalesed. The same numbering rules above for
// AbortReason apply here.
enum CompilerAbortReason {
    AbstractInterpreterInvalidType                =  10,
    ObjectAllocationSinkingAssertionFailure       = 100,
};

} // namespace JSC
