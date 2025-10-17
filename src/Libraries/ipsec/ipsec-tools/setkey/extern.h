/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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



void parse_init(void);
int parse(FILE **);
int parse_string(char *);

int setkeymsg(char *, size_t *);
int sendkeymsg(char *, size_t);

int yylex(void);
int yyparse(void);
void yyfatal(const char *);
void yyerror(const char *);

extern int f_rfcmode;
extern int lineno;
extern int last_msg_type;
extern u_int32_t last_priority;
extern int exit_now;

extern u_char m_buf[BUFSIZ];
extern u_int m_len;
extern int f_debug;

#ifdef HAVE_PFKEY_POLICY_PRIORITY
extern int last_msg_type;
extern u_int32_t last_priority;
#endif
