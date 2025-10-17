/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
/* $Id$ */

#ifndef __security_h__
#define __security_h__

enum protection_level {
    prot_invalid = -1,
    prot_clear = 0,
    prot_safe = 1,
    prot_confidential = 2,
    prot_private = 3
};

struct sec_client_mech {
    char *name;
    size_t size;
    int (*init)(void *);
    int (*auth)(void *, char*);
    void (*end)(void *);
    int (*check_prot)(void *, int);
    int (*overhead)(void *, int, int);
    int (*encode)(void *, void*, int, int, void**);
    int (*decode)(void *, void*, int, int);
};

struct sec_server_mech {
    char *name;
    size_t size;
    int (*init)(void *);
    void (*end)(void *);
    int (*check_prot)(void *, int);
    int (*overhead)(void *, int, int);
    int (*encode)(void *, void*, int, int, void**);
    int (*decode)(void *, void*, int, int);

    int (*auth)(void *);
    int (*adat)(void *, void*, size_t);
    size_t (*pbsz)(void *, size_t);
    int (*ccc)(void*);
    int (*userok)(void*, char*);
    int (*session)(void*, char*);
};

#define AUTH_OK		0
#define AUTH_CONTINUE	1
#define AUTH_ERROR	2

extern int ftp_do_gss_bindings;
extern int ftp_do_gss_delegate;
#ifdef FTP_SERVER
extern struct sec_server_mech krb4_server_mech, gss_server_mech;
#else
extern struct sec_client_mech krb4_client_mech, gss_client_mech;
#endif

extern int sec_complete;

#ifdef FTP_SERVER
extern char *ftp_command;
void new_ftp_command(char*);
void delete_ftp_command(void);
#endif

/* ---- */


int sec_fflush (FILE *);
int sec_fprintf (FILE *, const char *, ...)
    __attribute__ ((format (printf, 2,3)));
int sec_getc (FILE *);
int sec_putc (int, FILE *);
int sec_read (int, void *, int);
int sec_read_msg (char *, int);
int sec_vfprintf (FILE *, const char *, va_list)
    __attribute__ ((format (printf, 2,0)));
int sec_fprintf2(FILE *f, const char *fmt, ...)
    __attribute__ ((format (printf, 2,3)));
int sec_vfprintf2(FILE *, const char *, va_list)
    __attribute__ ((format (printf, 2,0)));
int sec_write (int, char *, int);

#ifdef FTP_SERVER
void adat (char *);
void auth (char *);
void ccc (void);
void mec (char *, enum protection_level);
void pbsz (int);
void prot (char *);
void delete_ftp_command (void);
void new_ftp_command (char *);
int sec_userok (char *);
int sec_session(char *);
int secure_command (void);
enum protection_level get_command_prot(void);
#else
void sec_end (void);
int sec_login (char *);
void sec_prot (int, char **);
void sec_prot_command (int, char **);
int sec_request_prot (char *);
void sec_set_protection_level (void);
void sec_status (void);

enum protection_level set_command_prot(enum protection_level);

#endif

#endif /* __security_h__ */
