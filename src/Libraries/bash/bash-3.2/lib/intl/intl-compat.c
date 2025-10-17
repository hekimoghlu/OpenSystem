/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "gettextP.h"

/* @@ end of prolog @@ */

/* This file redirects the gettext functions (without prefix) to those
   defined in the included GNU libintl library (with "libintl_" prefix).
   It is compiled into libintl in order to make the AM_GNU_GETTEXT test
   of gettext <= 0.11.2 work with the libintl library >= 0.11.3 which
   has the redirections primarily in the <libintl.h> include file.
   It is also compiled into libgnuintl so that libgnuintl.so can be used
   as LD_PRELOADable library on glibc systems, to provide the extra
   features that the functions in the libc don't have (namely, logging).  */


#undef gettext
#undef dgettext
#undef dcgettext
#undef ngettext
#undef dngettext
#undef dcngettext
#undef textdomain
#undef bindtextdomain
#undef bind_textdomain_codeset


/* When building a DLL, we must export some functions.  Note that because
   the functions are only defined for binary backward compatibility, we
   don't need to use __declspec(dllimport) in any case.  */
#if defined _MSC_VER && BUILDING_DLL
# define DLL_EXPORTED __declspec(dllexport)
#else
# define DLL_EXPORTED
#endif


DLL_EXPORTED
char *
gettext (msgid)
     const char *msgid;
{
  return libintl_gettext (msgid);
}


DLL_EXPORTED
char *
dgettext (domainname, msgid)
     const char *domainname;
     const char *msgid;
{
  return libintl_dgettext (domainname, msgid);
}


DLL_EXPORTED
char *
dcgettext (domainname, msgid, category)
     const char *domainname;
     const char *msgid;
     int category;
{
  return libintl_dcgettext (domainname, msgid, category);
}


DLL_EXPORTED
char *
ngettext (msgid1, msgid2, n)
     const char *msgid1;
     const char *msgid2;
     unsigned long int n;
{
  return libintl_ngettext (msgid1, msgid2, n);
}


DLL_EXPORTED
char *
dngettext (domainname, msgid1, msgid2, n)
     const char *domainname;
     const char *msgid1;
     const char *msgid2;
     unsigned long int n;
{
  return libintl_dngettext (domainname, msgid1, msgid2, n);
}


DLL_EXPORTED
char *
dcngettext (domainname, msgid1, msgid2, n, category)
     const char *domainname;
     const char *msgid1;
     const char *msgid2;
     unsigned long int n;
     int category;
{
  return libintl_dcngettext (domainname, msgid1, msgid2, n, category);
}


DLL_EXPORTED
char *
textdomain (domainname)
     const char *domainname;
{
  return libintl_textdomain (domainname);
}


DLL_EXPORTED
char *
bindtextdomain (domainname, dirname)
     const char *domainname;
     const char *dirname;
{
  return libintl_bindtextdomain (domainname, dirname);
}


DLL_EXPORTED
char *
bind_textdomain_codeset (domainname, codeset)
     const char *domainname;
     const char *codeset;
{
  return libintl_bind_textdomain_codeset (domainname, codeset);
}
