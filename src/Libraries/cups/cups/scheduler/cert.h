/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
 * Certificate structure...
 */

typedef struct cupsd_cert_s
{
  struct cupsd_cert_s *next;		/* Next certificate in list */
  int		pid;			/* Process ID (0 for root certificate) */
  char		certificate[33];	/* 32 hex characters, or 128 bits */
  char		username[33];		/* Authenticated username */
  int		type;			/* AuthType for username */
} cupsd_cert_t;


/*
 * Globals...
 */

VAR cupsd_cert_t	*Certs		/* List of certificates */
				VALUE(NULL);
VAR time_t		RootCertTime	/* Root certificate update time */
				VALUE(0);


/*
 * Prototypes...
 */

extern void		cupsdAddCert(int pid, const char *username, int type);
extern void		cupsdDeleteCert(int pid);
extern void		cupsdDeleteAllCerts(void);
extern cupsd_cert_t	*cupsdFindCert(const char *certificate);
extern void		cupsdInitCerts(void);
