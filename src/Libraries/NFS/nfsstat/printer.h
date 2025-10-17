/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
#ifndef printer_h
#define printer_h

#include <stdio.h>

#include <nfs/nfs.h>

#define PRINTER_NO_PREFIX NULL

struct nfsstats_printer {
	void  (*dump)(void);
	int   (*title)(const char * __restrict format, ...);
	void  (*newline)(void);
	void  (*open)(const char *prefix, const char *title);
	void  (*open_inside_array)(const char *prefix, const char *title);
	void  (*close)(void);
	void  (*open_array)(const char *prefix, const char *title, int *flags);
	void  (*add_array_str)(char sep, const char *flag, const char* value);
	void  (*add_array_num)(char sep, const char *flag, long value);
	void  (*close_array)(int opened_with_flags);
	void  (*open_locations)(const char *prefix, const char *title, uint32_t flags, uint32_t loc, uint32_t serv, uint32_t addr, int invalid);
	void  (*add_locations)(const char *path, const char *server, uint32_t addrcount, char **addrs);
	void  (*mount_header)(const char *to, const char *from);
	void  (*mount_fh)(uint32_t fh_len, unsigned char *fh_data);
	void  (*exports)(struct nfs_export_stat_rec  *rec);
	void  (*active_users)(struct nfs_user_stat_user_rec *rec, const char *addr, struct passwd *pw, int printuuid, time_t hr, time_t min, time_t sec);
	void  (*intpr)(int numargs, ...);
	void  (*nfserrs)(int numargs, ...);
};

/* Printf Printer */

void printf_null(void);
void printf_newline(void);
void printf_open(const char *, const char *);
void printf_open_array(const char *, const char *, int *);
void printf_add_array_str(char, const char *, const char *);
void printf_add_array_num(char, const char *, long);
void printf_close_array(int);
void printf_open_locations(const char *, const char *, uint32_t, uint32_t, uint32_t, uint32_t, int);
void printf_add_locations(const char *, const char *, uint32_t, char **);
void printf_mount_header(const char *, const char *);
void printf_mount_fh(uint32_t, unsigned char *);
void printf_exports(struct nfs_export_stat_rec *);
void printf_active_users(struct nfs_user_stat_user_rec *, const char *, struct passwd *, int, time_t, time_t, time_t);
void printf_intpr(int, ...);
void printf_nfserrs(int, ...);

static const struct nfsstats_printer printf_printer = {
	.dump              = printf_null,
	.title             = printf,
	.newline           = printf_newline,
	.open              = printf_open,
	.open_inside_array = printf_open,
	.close             = printf_null,
	.open_array        = printf_open_array,
	.add_array_str     = printf_add_array_str,
	.add_array_num     = printf_add_array_num,
	.close_array       = printf_close_array,
	.open_locations    = printf_open_locations,
	.add_locations     = printf_add_locations,
	.mount_header      = printf_mount_header,
	.mount_fh          = printf_mount_fh,
	.exports           = printf_exports,
	.active_users      = printf_active_users,
	.intpr             = printf_intpr,
	.nfserrs           = printf_nfserrs
};

/* Printf Printer */

/* Json Printer */

void json_dump(void);
int  json_title(const char * __restrict, ...);
void json_open_dictionary(const char *, const char *);
void json_open_dictionary_inside_array(const char *, const char *);
void json_close_dictionary(void);
void json_open_array(const char *, const char *, int *);
void json_add_array_str(char, const char *, const char *);
void json_add_array_num(char, const char *, long);
void json_close_array(int);
void json_open_locations(const char *, const char *, uint32_t, uint32_t, uint32_t, uint32_t, int);
void json_add_locations(const char *, const char *, uint32_t, char **);
void json_mount_header(const char *, const char *);
void json_mount_fh(uint32_t, unsigned char *);
void json_exports(struct nfs_export_stat_rec *);
void json_active_users(struct nfs_user_stat_user_rec *, const char *, struct passwd *, int, time_t, time_t, time_t);
void json_intpr(int, ...);
#define json_nfserrs json_intpr

static const struct nfsstats_printer json_printer = {
	.dump              = json_dump,
	.title             = json_title,
	.newline           = printf_null,
	.open              = json_open_dictionary,
	.open_inside_array = json_open_dictionary_inside_array,
	.close             = json_close_dictionary,
	.open_array        = json_open_array,
	.add_array_str     = json_add_array_str,
	.add_array_num     = json_add_array_num,
	.close_array       = json_close_array,
	.open_locations    = json_open_locations,
	.add_locations     = json_add_locations,
	.mount_header      = json_mount_header,
	.mount_fh          = json_mount_fh,
	.exports           = json_exports,
	.active_users      = json_active_users,
	.intpr             = json_intpr,
	.nfserrs           = json_nfserrs
};

#endif /* printer_h */
