/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
/* Copyright (C) 1997-2002 Free Software Foundation, Inc.

   This file is part of GNU Bash, the Bourne Again SHell.

   Bash is free software; you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 2, or (at your option) any later
   version.

   Bash is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   You should have received a copy of the GNU General Public License along
   with Bash; see the file COPYING.  If not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA */

#include <config.h>

#if defined (CAN_REDEFINE_GETENV)

#if defined (HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#include <bashansi.h>
#include <errno.h>
#include <shell.h>

#ifndef errno
extern int errno;
#endif

extern char **environ;

/* We supply our own version of getenv () because we want library
   routines to get the changed values of exported variables. */
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
 */static char *last_tempenv_value = (char *)NULL;

char *
getenv (name)
     const char *name;
{
  SHELL_VAR *var;

  if (name == 0 || *name == '\0')
    return ((char *)NULL);

  var = find_tempenv_variable ((char *)name);
  if (var)
    {
      FREE (last_tempenv_value);

      last_tempenv_value = value_cell (var) ? savestring (value_cell (var)) : (char *)NULL;
      return (last_tempenv_value);
    }
  else if (shell_variables)
    {
      var = find_variable ((char *)name);
      if (var && exported_p (var))
	return (value_cell (var));
    }
  else
    {
      register int i, len;
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
 */for (i = 0, len = strlen (name); environ[i]; i++)
	{
	  if ((STREQN (environ[i], name, len)) && (environ[i][len] == '='))
	    return (environ[i] + len + 1);
	}
    }

  return ((char *)NULL);
}

/* Some versions of Unix use _getenv instead. */
char *
_getenv (name)
     const char *name;
{
  return (getenv (name));
}

/* SUSv3 says argument is a `char *'; BSD implementations disagree */
int
putenv (str)
#ifndef HAVE_STD_PUTENV
     const char *str;
#else
     char *str;
#endif
{
  SHELL_VAR *var;
  char *name, *value;
  int offset;

  if (str == 0 || *str == '\0')
    {
      errno = EINVAL;
      return -1;
    }

  offset = assignment (str, 0);
  if (str[offset] != '=')
    {
      errno = EINVAL;
      return -1;
    }
  name = savestring (str);
  name[offset] = 0;

  value = name + offset + 1;

  /* XXX - should we worry about readonly here? */
  var = bind_variable (name, value, 0);
  if (var == 0)
    {
      errno = EINVAL;
      return -1;
    }

  VUNSETATTR (var, att_invisible);
  VSETATTR (var, att_exported);

  return 0;
}

#if 0
int
_putenv (name)
#ifndef HAVE_STD_PUTENV
     const char *name;
#else
     char *name;
#endif
{
  return putenv (name);
}
#endif

int
setenv (name, value, rewrite)
     const char *name;
     const char *value;
     int rewrite;
{
  SHELL_VAR *var;
  char *v;

  if (name == 0 || *name == '\0' || strchr (name, '=') != 0)
    {
      errno = EINVAL;
      return -1;
    }

  var = 0;
  v = (char *)value;	/* some compilers need explicit cast */
  /* XXX - should we worry about readonly here? */
  if (rewrite == 0)
    var = find_variable (name);

  if (var == 0)
    var = bind_variable (name, v, 0);

  if (var == 0)
    return -1;

  VUNSETATTR (var, att_invisible);
  VSETATTR (var, att_exported);

  return 0;
}

#if 0
int
_setenv (name, value, rewrite)
     const char *name;
     const char *value;
     int rewrite;
{
  return setenv (name, value, rewrite);
}
#endif
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
 */#ifdef HAVE_STD_UNSETENV
#define UNSETENV_RETURN(N)	return(N)
#define UNSETENV_RETTYPE	int
#else
#define UNSETENV_RETURN(N)	return
#define UNSETENV_RETTYPE	void
#endif

UNSETENV_RETTYPE
unsetenv (name)
     const char *name;
{
  if (name == 0 || *name == '\0' || strchr (name, '=') != 0)
    {
      errno = EINVAL;
      UNSETENV_RETURN(-1);
    }

  /* XXX - should we just remove the export attribute here? */
#if 1
  unbind_variable (name);
#else
  SHELL_VAR *v;

  v = find_variable (name);
  if (v)
    VUNSETATTR (v, att_exported);
#endif

  UNSETENV_RETURN(0);
}
#endif /* CAN_REDEFINE_GETENV */

