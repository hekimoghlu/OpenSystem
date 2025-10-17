/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
#include <tcl.h>

/*
 * Exported tcl level procedures.
 *
 * ATTENTION:
 * due to the fact that cpp - processing with gcc 2.5.8 removes any comments
 * in macro-arguments (even if called with option '-C') i have to use the
 * predefined macro __C2MAN__ to distinguish real compilation and manpage
 * generation, removing _ANSI_ARGS_ in the latter case.
 */

/*
 * Replaces an entry for an existing channel.
 * Replaces an entry for an existing channel in both the global list
 * of channels and the hashtable of channels for the given
 * interpreter. The replacement is a new channel with same name, it
 * supercedes the replaced channel. From now on both input and output
 * of the superceded channel will go through the newly created
 * channel, thus allowing the arbitrary filtering/manipulation of the
 * data. It is the responsibility of the newly created channel to
 * forward the filtered/manipulated data to the channel he supercedes
 * at his leisure. The result of the command is the token for the new
 * channel.
 */

Tcl_Channel
Tcl_ReplaceChannel (Tcl_Interp *interp /* An interpreter having access
					* to the channel to supercede,
					* see 'prevChan' */,
		    Tcl_ChannelType *typePtr /* The channel type record
					      * for the new channel. */,
		    ClientData instanceData /* Instance specific data
					     * for the new channel. */,
		    int mask /* TCL_READABLE & TCL_WRITABLE to
			      * indicate whether the new channel
			      * should be readable and/or
			      * writable. This mask is mixed (by &)
			      * with the same information from the
			      * superceded channel to prevent the
			      * execution of invalid operations */,
		    Tcl_Channel prevChan /* The token of the channel to
					  * replace */);
/*
 * This is the reverse operation to 'Tcl_ReplaceChannel'.
 * This is the reverse operation to 'Tcl_ReplaceChannel'. It takes the
 * given channel and uncovers a superceded channel. If there is no
 * superceded channel this operation is equivalent to 'Tcl_Close'. The
 * superceding channel is destroyed.
 */

void
Tcl_UndoReplaceChannel (Tcl_Interp *interp /* An interpreter having access
					    * to the channel to unstack,
					    * see 'chan' */,
			Tcl_Channel chan /* The channel to remove */);

