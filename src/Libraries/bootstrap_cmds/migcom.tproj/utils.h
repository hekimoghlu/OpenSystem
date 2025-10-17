/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
/*
 * HISTORY
 * 07-Apr-89  Richard Draves (rpd) at Carnegie-Mellon University
 * Extensive revamping.  Added polymorphic arguments.
 * Allow multiple variable-sized inline arguments in messages.
 *
 * 28-May-87  Richard Draves (rpd) at Carnegie-Mellon University
 * Created.
 */

#ifndef _UTILS_H
#define _UTILS_H

/* definitions used by the Event Logger */

#define LOG_USER 0
#define LOG_SERVER 1

#define LOG_REQUEST 0
#define LOG_REPLY 1

/* stuff used by more than one of header.c, user.c, server.c */

extern void WriteMigExternal(FILE *file);
extern void WriteMigInternal(FILE *file);

extern void WriteImport(FILE *file, string_t filename);
extern void WriteRCSDecl( FILE *file, identifier_t name, string_t rcs );
extern void WriteBogusDefines( FILE *file );

extern void WriteList( FILE *file, argument_t *args,
  void (*func)(FILE *file, argument_t *arg),
  u_int mask, char *between, char *after );

extern void WriteReverseList( FILE *file, argument_t *args,
 void (*func)(FILE *file, argument_t *arg),
 u_int mask, char *between, char *after );

/* good as arguments to WriteList */
extern void WriteNameDecl( FILE *file, argument_t *arg );
extern void WriteUserVarDecl( FILE *file, argument_t *arg );
extern void WriteServerVarDecl( FILE *file, argument_t *arg );
extern void WriteTemplateDeclIn( FILE *file, argument_t *arg );
extern void WriteTemplateDeclOut( FILE *file, argument_t *arg );
extern void WriteCheckDecl( FILE *file, argument_t *arg );

extern char *ReturnTypeStr( routine_t *rt );

extern char *FetchUserType( ipc_type_t *it );
extern char *FetchServerType( ipc_type_t *it );
extern char *FetchKPDType( ipc_type_t *it );
extern void WriteKPDFieldDecl(FILE *file, argument_t *arg);

extern void WriteFieldDeclPrim( FILE *file, argument_t *arg, char *(*tfunc)(ipc_type_t *it) );

extern void WriteStructDecl( FILE *file, argument_t *args,
        void (*func)(FILE *file, argument_t *arg),
        u_int mask, char *name,
        boolean_t simple, boolean_t trailer,
        boolean_t isuser,
        boolean_t template_only );

extern void WriteUDataStructDecl(
        FILE *file,
        argument_t *args,
        void (*func)(FILE *, argument_t *),
        u_int mask,
        char *name,
        boolean_t simple,
        boolean_t trailer,
        boolean_t trailer_t,
        boolean_t template_only);

extern void WriteStaticDecl( FILE *file, argument_t *arg );

extern void WriteCopyType(FILE *file, ipc_type_t *it, boolean_t mig_allocated_buf, char *left, char *right, ...);

extern void WriteCopyArg(FILE *file, argument_t *arg, boolean_t mig_allocated_buf, char *left, char *right, ...);

extern void WriteLogMsg( FILE *file, routine_t *rt, int where, int what );

extern void WriteCheckTrailerHead( FILE *file, routine_t *rt, boolean_t isuser );

extern void WriteCheckTrailerSize( FILE *file, boolean_t isuser, argument_t *arg );

extern void WriteReturnMsgError( FILE *file, routine_t *rt, boolean_t isuser, argument_t *arg, string_t error );

extern void WriteRPCRoutineArgDescriptor( FILE *file, routine_t *rt );

extern void WriteRequestTypes( FILE *file, statement_t *stats);
extern void WriteCheckRequests( FILE *file, statement_t *stats);
extern void WriteUserRequestUnion( FILE *file, statement_t *stats );
extern void WriteServerRequestUnion( FILE *file, statement_t *stats );

extern void WriteReplyTypes( FILE *file, statement_t *stats);
extern void WriteCheckReplies( FILE *file, statement_t *stats);
extern void WriteUserReplyUnion( FILE *file, statement_t *stats );
extern void WriteServerReplyUnion( FILE *file, statement_t *stats );

extern void WriteNDRConvertArgDecl( FILE *file, argument_t *arg, char *convert, char *dir);

#endif /* _UTILS_H */
