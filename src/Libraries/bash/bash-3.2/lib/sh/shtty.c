/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
 * shtty.c -- abstract interface to the terminal, focusing on capabilities.
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif

#include <shtty.h>

static TTYSTRUCT ttin, ttout;
static int ttsaved = 0;

int
ttgetattr(fd, ttp)
int	fd;
TTYSTRUCT *ttp;
{
#ifdef TERMIOS_TTY_DRIVER
  return tcgetattr(fd, ttp);
#else
#  ifdef TERMIO_TTY_DRIVER
  return ioctl(fd, TCGETA, ttp);
#  else
  return ioctl(fd, TIOCGETP, ttp);
#  endif
#endif
}

int
ttsetattr(fd, ttp)
int	fd;
TTYSTRUCT *ttp;
{
#ifdef TERMIOS_TTY_DRIVER
  return tcsetattr(fd, TCSADRAIN, ttp);
#else
#  ifdef TERMIO_TTY_DRIVER
  return ioctl(fd, TCSETAW, ttp);
#  else
  return ioctl(fd, TIOCSETN, ttp);
#  endif
#endif
}

void
ttsave()
{
  if (ttsaved)
   return;
  ttgetattr (0, &ttin);
  ttgetattr (1, &ttout);
  ttsaved = 1;
}

void
ttrestore()
{
  if (ttsaved == 0)
    return;
  ttsetattr (0, &ttin);
  ttsetattr (1, &ttout);
  ttsaved = 0;
}

/* Retrieve the attributes associated with tty fd FD. */
TTYSTRUCT *
ttattr (fd)
     int fd;
{
  if (ttsaved == 0)
    return ((TTYSTRUCT *)0);
  if (fd == 0)
    return &ttin;
  else if (fd == 1)
    return &ttout;
  else
    return ((TTYSTRUCT *)0);
}

/*
 * Change attributes in ttp so that when it is installed using
 * ttsetattr, the terminal will be in one-char-at-a-time mode.
 */
int
tt_setonechar(ttp)
     TTYSTRUCT *ttp;
{
#if defined (TERMIOS_TTY_DRIVER) || defined (TERMIO_TTY_DRIVER)

  /* XXX - might not want this -- it disables erase and kill processing. */
  ttp->c_lflag &= ~ICANON;

  ttp->c_lflag |= ISIG;
#  ifdef IEXTEN
  ttp->c_lflag |= IEXTEN;
#  endif

  ttp->c_iflag |= ICRNL;	/* make sure we get CR->NL on input */
  ttp->c_iflag &= ~INLCR;	/* but no NL->CR */

#  ifdef OPOST
  ttp->c_oflag |= OPOST;
#  endif
#  ifdef ONLCR
  ttp->c_oflag |= ONLCR;
#  endif
#  ifdef OCRNL
  ttp->c_oflag &= ~OCRNL;
#  endif
#  ifdef ONOCR
  ttp->c_oflag &= ~ONOCR;
#  endif
#  ifdef ONLRET
  ttp->c_oflag &= ~ONLRET;
#  endif

  ttp->c_cc[VMIN] = 1;
  ttp->c_cc[VTIME] = 0;

#else

  ttp->sg_flags |= CBREAK;

#endif

  return 0;
}

/* Set the terminal into one-character-at-a-time mode */
int
ttonechar ()
{
  TTYSTRUCT tt;

  if (ttsaved == 0)
    return -1;
  tt = ttin;
  if (tt_setonechar(&tt) < 0)
    return -1;
  return (ttsetattr (0, &tt));
}

/*
 * Change attributes in ttp so that when it is installed using
 * ttsetattr, the terminal will be in no-echo mode.
 */
int
tt_setnoecho(ttp)
     TTYSTRUCT *ttp;
{
#if defined (TERMIOS_TTY_DRIVER) || defined (TERMIO_TTY_DRIVER)
  ttp->c_lflag &= ~(ECHO|ECHOK|ECHONL);
#else
  ttp->sg_flags &= ~ECHO;
#endif

  return 0;
}

/* Set the terminal into no-echo mode */
int
ttnoecho ()
{
  TTYSTRUCT tt;

  if (ttsaved == 0)
    return -1;
  tt = ttin;
  if (tt_setnoecho (&tt) < 0)
    return -1;
  return (ttsetattr (0, &tt));
}

/*
 * Change attributes in ttp so that when it is installed using
 * ttsetattr, the terminal will be in eight-bit mode (pass8).
 */
int
tt_seteightbit (ttp)
     TTYSTRUCT *ttp;
{
#if defined (TERMIOS_TTY_DRIVER) || defined (TERMIO_TTY_DRIVER)
  ttp->c_iflag &= ~ISTRIP;
  ttp->c_cflag |= CS8;
  ttp->c_cflag &= ~PARENB;
#else
  ttp->sg_flags |= ANYP;
#endif

  return 0;
}

/* Set the terminal into eight-bit mode */
int
tteightbit ()
{
  TTYSTRUCT tt;

  if (ttsaved == 0)
    return -1;
  tt = ttin;
  if (tt_seteightbit (&tt) < 0)
    return -1;
  return (ttsetattr (0, &tt));
}

/*
 * Change attributes in ttp so that when it is installed using
 * ttsetattr, the terminal will be in non-canonical input mode.
 */
int
tt_setnocanon (ttp)
     TTYSTRUCT *ttp;
{
#if defined (TERMIOS_TTY_DRIVER) || defined (TERMIO_TTY_DRIVER)
  ttp->c_lflag &= ~ICANON;
#endif

  return 0;
}

/* Set the terminal into non-canonical mode */
int
ttnocanon ()
{
  TTYSTRUCT tt;

  if (ttsaved == 0)
    return -1;
  tt = ttin;
  if (tt_setnocanon (&tt) < 0)
    return -1;
  return (ttsetattr (0, &tt));
}

/*
 * Change attributes in ttp so that when it is installed using
 * ttsetattr, the terminal will be in cbreak, no-echo mode.
 */
int
tt_setcbreak(ttp)
     TTYSTRUCT *ttp;
{
  if (tt_setonechar (ttp) < 0)
    return -1;
  return (tt_setnoecho (ttp));
}

/* Set the terminal into cbreak (no-echo, one-character-at-a-time) mode */
int
ttcbreak ()
{
  TTYSTRUCT tt;

  if (ttsaved == 0)
    return -1;
  tt = ttin;
  if (tt_setcbreak (&tt) < 0)
    return -1;
  return (ttsetattr (0, &tt));
}
