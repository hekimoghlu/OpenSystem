/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#ifndef _RDE_DS_TC_H
#define _RDE_DS_TC_H 1

#include <util.h> /* Scoping */

typedef struct RDE_TC_* RDE_TC;

/* SKIP START */
SCOPE RDE_TC rde_tc_new  (void);
SCOPE void   rde_tc_del  (RDE_TC tc);

SCOPE long int    rde_tc_size   (RDE_TC tc);
SCOPE void        rde_tc_clear  (RDE_TC tc);
SCOPE char*       rde_tc_append (RDE_TC tc, char* ch, long int len);
SCOPE void        rde_tc_get    (RDE_TC tc, int at, char** ch, long int *len);
SCOPE void        rde_tc_get_s  (RDE_TC tc, int at, int last, char** ch, long int *len);
/* SKIP END */
#endif /* _RDE_DS_TC_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
