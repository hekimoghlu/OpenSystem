/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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

#define APPMACROS_ONLY
#include "applink.c"

extern void *OPENSSL_UplinkTable[];

#define UP_stdin  (*(void *(*)(void))OPENSSL_UplinkTable[APPLINK_STDIN])()
#define UP_stdout (*(void *(*)(void))OPENSSL_UplinkTable[APPLINK_STDOUT])()
#define UP_stderr (*(void *(*)(void))OPENSSL_UplinkTable[APPLINK_STDERR])()
#define UP_fprintf (*(int (*)(void *,const char *,...))OPENSSL_UplinkTable[APPLINK_FPRINTF])
#define UP_fgets  (*(char *(*)(char *,int,void *))OPENSSL_UplinkTable[APPLINK_FGETS])
#define UP_fread  (*(size_t (*)(void *,size_t,size_t,void *))OPENSSL_UplinkTable[APPLINK_FREAD])
#define UP_fwrite (*(size_t (*)(const void *,size_t,size_t,void *))OPENSSL_UplinkTable[APPLINK_FWRITE])
#define UP_fsetmod (*(int (*)(void *,char))OPENSSL_UplinkTable[APPLINK_FSETMOD])
#define UP_feof   (*(int (*)(void *))OPENSSL_UplinkTable[APPLINK_FEOF])
#define UP_fclose (*(int (*)(void *))OPENSSL_UplinkTable[APPLINK_FCLOSE])

#define UP_fopen  (*(void *(*)(const char *,const char *))OPENSSL_UplinkTable[APPLINK_FOPEN])
#define UP_fseek  (*(int (*)(void *,long,int))OPENSSL_UplinkTable[APPLINK_FSEEK])
#define UP_ftell  (*(long (*)(void *))OPENSSL_UplinkTable[APPLINK_FTELL])
#define UP_fflush (*(int (*)(void *))OPENSSL_UplinkTable[APPLINK_FFLUSH])
#define UP_ferror (*(int (*)(void *))OPENSSL_UplinkTable[APPLINK_FERROR])
#define UP_clearerr (*(void (*)(void *))OPENSSL_UplinkTable[APPLINK_CLEARERR])
#define UP_fileno (*(int (*)(void *))OPENSSL_UplinkTable[APPLINK_FILENO])

#define UP_open   (*(int (*)(const char *,int,...))OPENSSL_UplinkTable[APPLINK_OPEN])
#define UP_read   (*(ssize_t (*)(int,void *,size_t))OPENSSL_UplinkTable[APPLINK_READ])
#define UP_write  (*(ssize_t (*)(int,const void *,size_t))OPENSSL_UplinkTable[APPLINK_WRITE])
#define UP_lseek  (*(long (*)(int,long,int))OPENSSL_UplinkTable[APPLINK_LSEEK])
#define UP_close  (*(int (*)(int))OPENSSL_UplinkTable[APPLINK_CLOSE])
