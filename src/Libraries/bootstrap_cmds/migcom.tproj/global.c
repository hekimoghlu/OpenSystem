/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
/*
 * Mach Operating System
 * Copyright (c) 1991,1990 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 */

#include "strdefs.h"
#include "global.h"
#include "error.h"
#include "mig_machine.h"

boolean_t PrintVersion = FALSE;
boolean_t BeQuiet = FALSE;
boolean_t BeVerbose = FALSE;
boolean_t UseMsgRPC = TRUE;
boolean_t GenSymTab = FALSE;
boolean_t UseEventLogger = FALSE;
boolean_t BeLint = FALSE;
boolean_t BeAnsiC = TRUE;
boolean_t CheckNDR = FALSE;
boolean_t PackMsg = PACK_MESSAGES;
boolean_t UseSplitHeaders = FALSE;
boolean_t ShortCircuit = FALSE;
boolean_t UseRPCTrap = FALSE;
boolean_t TestRPCTrap= FALSE;
boolean_t IsVoucherCodeAllowed = TRUE;
boolean_t EmitCountAnnotations = FALSE;

boolean_t IsKernelUser = FALSE;
boolean_t IsKernelServer = FALSE;
boolean_t UseMachMsg2 = FALSE;
boolean_t UseSpecialReplyPort = FALSE;
boolean_t HasUseSpecialReplyPort = FALSE;
boolean_t HasConsumeOnSendError = FALSE;
u_int ConsumeOnSendError = 0;
int MaxServerDescrs = -1;
int MaxServerReplyDescrs = -1;

string_t RCSId = strNULL;

string_t SubsystemName = strNULL;
u_int SubsystemBase = 0;

string_t MsgOption = strNULL;
string_t WaitTime = strNULL;
string_t SendTime = strNULL;
string_t ErrorProc = "MsgError";
string_t ServerPrefix = "";
string_t UserPrefix = "";
string_t ServerDemux = strNULL;
string_t ServerImpl = strNULL;
string_t ServerSubsys = strNULL;
int MaxMessSizeOnStack = -1;    /* by default, always on stack */
int UserTypeLimit = -1;         /* by default, assume unlimited size. */

string_t yyinname;

char NewCDecl[] = "(defined(__STDC__) || defined(c_plusplus))";
char LintLib[] = "defined(LINTLIBRARY)";

void
init_global()
{
    yyinname = strmake("<no name yet>");
}

string_t UserFilePrefix = strNULL;
string_t UserHeaderFileName = strNULL;
string_t ServerHeaderFileName = strNULL;
string_t InternalHeaderFileName = strNULL;
string_t DefinesHeaderFileName = strNULL;
string_t UserFileName = strNULL;
string_t ServerFileName = strNULL;
string_t GenerationDate = strNULL;

void
more_global()
{
  if (SubsystemName == strNULL)
    fatal("no SubSystem declaration");

  if (UserHeaderFileName == strNULL)
    UserHeaderFileName = strconcat(SubsystemName, ".h");
  else if (streql(UserHeaderFileName, "/dev/null"))
    UserHeaderFileName = strNULL;

  if (UserFileName == strNULL)
    UserFileName = strconcat(SubsystemName, "User.c");
  else if (streql(UserFileName, "/dev/null"))
    UserFileName = strNULL;

  if (ServerFileName == strNULL)
    ServerFileName = strconcat(SubsystemName, "Server.c");
  else if (streql(ServerFileName, "/dev/null"))
    ServerFileName = strNULL;

  if (ServerDemux == strNULL)
    ServerDemux = strconcat(SubsystemName, "_server");

  if (ServerImpl == strNULL)
    ServerImpl = strconcat(SubsystemName, "_impl");

  if (ServerSubsys == strNULL) {
    if (ServerPrefix != strNULL)
      ServerSubsys = strconcat(ServerPrefix, SubsystemName);
    else
      ServerSubsys = SubsystemName;
    ServerSubsys = strconcat(ServerSubsys, "_subsystem");
  }
  if (HasUseSpecialReplyPort && !BeAnsiC) {
    fatal("Cannot use UseSpecialReplyPort in non ANSI mode\n");
  }
}
