/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
#ifndef GENWRAP_H
#define	GENWRAP_H

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>

/* Parser bits */
void yyerror(const char *s);
int yylex(void);
int yyparse(void);

extern FILE		*yyin;
extern const char	*yyfile;
extern int		yyline;

/* Application logic */
struct app;
#define	ARGFLAG_LOGONLY	0x0001

/* Don't add these flags to aliases. */
#define	ARGFLAG_NO_ALIAS	(ARGFLAG_LOGONLY)

struct app *app_add(struct app *current_app, const char *name);
void app_set_default(struct app *app);
void app_set_argmode_logonly(struct app *app);
void app_add_addarg(struct app *app, const char **argv, int nargv);
void app_set_path(struct app *app, const char *path, bool relcwd);
const char *app_get_path(const struct app *app);
void app_add_flag(struct app *app, const char *flag, const char *alias,
    int argument, uint32_t flags, const char *pattern);

void wrapper_set_analytics(const char *id, bool noargs);
void wrapper_set_envvar(const char *var);

#endif	/* GENWRAP_H */
