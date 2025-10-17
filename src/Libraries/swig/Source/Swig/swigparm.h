/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
/* $Id: swig.h 9629 2006-12-30 18:27:47Z beazley $ */

/* Individual parameters */
extern Parm      *NewParm(SwigType *type, const_String_or_char_ptr name);
extern Parm      *CopyParm(Parm *p);

/* Parameter lists */
extern ParmList  *CopyParmList(ParmList *);
extern ParmList  *CopyParmListMax(ParmList *, int count);
extern int        ParmList_len(ParmList *);
extern int        ParmList_numrequired(ParmList *);
extern int        ParmList_has_defaultargs(ParmList *p);

/* Output functions */
extern String    *ParmList_str(ParmList *);
extern String    *ParmList_str_defaultargs(ParmList *);
extern String    *ParmList_protostr(ParmList *);


