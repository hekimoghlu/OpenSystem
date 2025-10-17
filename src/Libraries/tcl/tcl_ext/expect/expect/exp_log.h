/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
extern void		expErrorLog _ANSI_ARGS_(TCL_VARARGS(char *,fmt));
extern void		expErrorLogU _ANSI_ARGS_((char *));

extern void		expStdoutLog _ANSI_ARGS_(TCL_VARARGS(int,force_stdout));
extern void		expStdoutLogU _ANSI_ARGS_((char *buf, int force_stdout));

EXTERN void		expDiagInit _ANSI_ARGS_((void));
EXTERN int		expDiagChannelOpen _ANSI_ARGS_((Tcl_Interp *,char *));
EXTERN Tcl_Channel	expDiagChannelGet _ANSI_ARGS_((void));
EXTERN void		expDiagChannelClose _ANSI_ARGS_((Tcl_Interp *));
EXTERN char *		expDiagFilename _ANSI_ARGS_((void));
EXTERN int		expDiagToStderrGet _ANSI_ARGS_((void));
EXTERN void		expDiagToStderrSet _ANSI_ARGS_((int));
EXTERN void		expDiagWriteBytes _ANSI_ARGS_((char *,int));
EXTERN void		expDiagWriteChars _ANSI_ARGS_((char *,int));
EXTERN void		expDiagWriteObj _ANSI_ARGS_((Tcl_Obj *));
EXTERN void		expDiagLog _ANSI_ARGS_(TCL_VARARGS(char *,fmt));
EXTERN void		expDiagLogU _ANSI_ARGS_((char *));

EXTERN char *		expPrintify _ANSI_ARGS_((char *));
EXTERN char *		expPrintifyUni _ANSI_ARGS_((Tcl_UniChar *,int));
EXTERN char *		expPrintifyObj _ANSI_ARGS_((Tcl_Obj *));
EXTERN void		expPrintf _ANSI_ARGS_(TCL_VARARGS(char *,fmt));

EXTERN void		expLogInit _ANSI_ARGS_((void));
EXTERN int		expLogChannelOpen _ANSI_ARGS_((Tcl_Interp *,char *,int));
EXTERN Tcl_Channel 	expLogChannelGet _ANSI_ARGS_((void));
EXTERN int		expLogChannelSet _ANSI_ARGS_((Tcl_Interp *,char *));
EXTERN void		expLogChannelClose _ANSI_ARGS_((Tcl_Interp *));
EXTERN char *		expLogFilenameGet _ANSI_ARGS_((void));
EXTERN void		expLogAppendSet _ANSI_ARGS_((int));
EXTERN int		expLogAppendGet _ANSI_ARGS_((void));
EXTERN void		expLogLeaveOpenSet _ANSI_ARGS_((int));
EXTERN int		expLogLeaveOpenGet _ANSI_ARGS_((void));
EXTERN void		expLogAllSet _ANSI_ARGS_((int));
EXTERN int		expLogAllGet _ANSI_ARGS_((void));
EXTERN void		expLogToStdoutSet _ANSI_ARGS_((int));
EXTERN int		expLogToStdoutGet _ANSI_ARGS_((void));
EXTERN void		expLogDiagU _ANSI_ARGS_((char *));
EXTERN int		expWriteBytesAndLogIfTtyU _ANSI_ARGS_((ExpState *,Tcl_UniChar *,int));

EXTERN int		expLogUserGet _ANSI_ARGS_((void));
EXTERN void		expLogUserSet _ANSI_ARGS_((int));

EXTERN void		expLogInteractionU _ANSI_ARGS_((ExpState *,Tcl_UniChar *,int));
