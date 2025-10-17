/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#include "login_locl.h"

RCSID("$Id$");

#ifdef HAVE_SHADOW_H

#ifndef _PATH_CHPASS
#define _PATH_CHPASS "/usr/bin/passwd"
#endif

static int
change_passwd(const struct passwd *who)
{
    int status;
    pid_t pid;

    switch (pid = fork()) {
    case -1:
        printf("fork /bin/passwd");
        exit(1);
    case 0:
        execlp(_PATH_CHPASS, "passwd", who->pw_name, (char *) 0);
        exit(1);
    default:
        waitpid(pid, &status, 0);
        return (status);
    }
}

void
check_shadow(const struct passwd *pw, const struct spwd *sp)
{
  long today;

  today = time(0)/(24L * 60 * 60);

  if (sp == NULL)
      return;

  if (sp->sp_expire > 0) {
        if (today >= sp->sp_expire) {
            printf("Your account has expired.\n");
            sleep(1);
            exit(0);
        } else if (sp->sp_expire - today < 14) {
            printf("Your account will expire in %d days.\n",
                   (int)(sp->sp_expire - today));
        }
  }

  if (sp->sp_max > 0) {
        if (today >= (sp->sp_lstchg + sp->sp_max)) {
            printf("Your password has expired. Choose a new one.\n");
            change_passwd(pw);
        } else if (sp->sp_warn > 0
            && (today > (sp->sp_lstchg + sp->sp_max - sp->sp_warn))) {
            printf("Your password will expire in %d days.\n",
                   (int)(sp->sp_lstchg + sp->sp_max - today));
        }
  }
}
#endif /* HAVE_SHADOW_H */
