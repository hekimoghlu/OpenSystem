/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#define PERL_NO_GET_CONTEXT
#include "EXTERN.h"  
#include "perl.h"
#include "XSUB.h"

#include <db.h>

void
#ifdef CAN_PROTOTYPE
__getBerkeleyDBInfo(void)
#else
__getBerkeleyDBInfo()
#endif
{
#ifdef dTHX	
    dTHX;
#endif    
    SV * version_sv = perl_get_sv("DB_File::db_version", GV_ADD|GV_ADDMULTI) ;
    SV * ver_sv = perl_get_sv("DB_File::db_ver", GV_ADD|GV_ADDMULTI) ;
    SV * compat_sv = perl_get_sv("DB_File::db_185_compat", GV_ADD|GV_ADDMULTI) ;

#ifdef DB_VERSION_MAJOR
    int Major, Minor, Patch ;

    (void)db_version(&Major, &Minor, &Patch) ;

    /* Check that the versions of db.h and libdb.a are the same */
    if (Major != DB_VERSION_MAJOR || Minor != DB_VERSION_MINOR )
		/* || Patch != DB_VERSION_PATCH) */

	croak("\nDB_File was build with libdb version %d.%d.%d,\nbut you are attempting to run it with libdb version %d.%d.%d\n",
		DB_VERSION_MAJOR, DB_VERSION_MINOR, DB_VERSION_PATCH, 
		Major, Minor, Patch) ;
    
    /* check that libdb is recent enough  -- we need 2.3.4 or greater */
    if (Major == 2 && (Minor < 3 || (Minor ==  3 && Patch < 4)))
	croak("DB_File needs Berkeley DB 2.3.4 or greater, you have %d.%d.%d\n",
		 Major, Minor, Patch) ;
 
    {
        char buffer[40] ;
        sprintf(buffer, "%d.%d", Major, Minor) ;
        sv_setpv(version_sv, buffer) ; 
        sprintf(buffer, "%d.%03d%03d", Major, Minor, Patch) ;
        sv_setpv(ver_sv, buffer) ; 
    }
 
#else /* ! DB_VERSION_MAJOR */
    sv_setiv(version_sv, 1) ;
    sv_setiv(ver_sv, 1) ;
#endif /* ! DB_VERSION_MAJOR */

#ifdef COMPAT185
    sv_setiv(compat_sv, 1) ;
#else /* ! COMPAT185 */
    sv_setiv(compat_sv, 0) ;
#endif /* ! COMPAT185 */

}
