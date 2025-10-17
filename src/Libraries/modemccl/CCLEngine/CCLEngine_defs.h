/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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
#ifndef __CCLENGINE_DEFS__
#define __CCLENGINE_DEFS__


enum {

    cclErr_AbortMatchRead = -6000,	// internal error used to abort match read
    cclErr_BadParameter = -6001,	// Bad parameter given to the engine
    cclErr_DuplicateLabel = -6002,	// Duplicate label
    cclErr_LabelUndefined = -6003,	// Label undefined
    cclErr_SubroutineOverFlow = -6004,	// Subroutine overflow
    cclErr_NoMemErr = -6005,		// No memory...
    cclErr = -6006,			// CCL error base
    cclErr_CloseError = -6007,		// There is at least one script open
    cclErr_ScriptCancelled = -6008,	// Script Canceled
    cclErr_TooManyLines = -6009,	// Script contains too many lines
    cclErr_ScriptTooBig = -6010,	// Script contains too many characters
    cclErr_NotInitialized = -6011,	// CCL has not been initialized
    cclErr_CancelInProgress = -6012,	// Cancel in progress.
    cclErr_PlayInProgress = -6013,	// Play command already in progress.
    cclErr_ExitOK = -6014,	 	// Exit with no error.
    cclErr_BadLabel = -6015,		// Label out of range.
    cclErr_BadCommand = -6016,		// Bad command.
    cclErr_EndOfScriptErr = -6017,	// End of script reached, expecting Exit.
    cclErr_MatchStrIndxErr = -6018,	// Match string index is out of bounds.
    cclErr_ModemErr = -6019, 		// Modem error, modem not responding.
    cclErr_NoDialTone = -6020,		// No dial tone.
    cclErr_NoCarrierErr = -6021, 	// No carrier.
    cclErr_LineBusyErr = -6022, 	// Line busy.
    cclErr_NoAnswerErr = -6023,		// No answer.
    cclErr_NoOriginateLabel = -6024,	// No @ORIGINATE label
    cclErr_NoAnswerLabel = -6025,	// No @ANSWER label
    cclErr_NoHangUpLabel = -6026,	// No @HANGUP label
    cclErr_NoNumberErr = -6027,  	// Can't connect because number is empty.
    cclErr_BadScriptErr = -6028  	// Incorrect script for the modem.
};

#endif	/* __CCLENGINE_DEFS__ */