/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 3, 2025.
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
 * This is the overlay structure. It is used to create a seperate
 * layer over the current windows.
 */

struct mchar;	/* forward declaration */

struct LayFuncs
{
  void	(*lf_LayProcess) __P((char **, int *));
  void	(*lf_LayAbort) __P((void));
  void	(*lf_LayRedisplayLine) __P((int, int, int, int));
  void	(*lf_LayClearLine) __P((int, int, int, int));
  int	(*lf_LayRewrite) __P((int, int, int, struct mchar *, int));
  int	(*lf_LayResize) __P((int, int));
  void	(*lf_LayRestore) __P((void));
};

struct layer
{
  struct canvas *l_cvlist;	/* list of canvases displaying layer */
  int	 l_width;
  int	 l_height;
  int    l_x;			/* cursor position */
  int    l_y;
  int    l_encoding;
  struct LayFuncs *l_layfn;
  char	*l_data;

  struct layer *l_next;		/* layer stack, should be in data? */
  struct layer *l_bottom;	/* bottom element of layer stack */
  int	 l_blocking;
};

#define LayProcess		(*flayer->l_layfn->lf_LayProcess)
#define LayAbort		(*flayer->l_layfn->lf_LayAbort)
#define LayRedisplayLine	(*flayer->l_layfn->lf_LayRedisplayLine)
#define LayClearLine		(*flayer->l_layfn->lf_LayClearLine)
#define LayRewrite		(*flayer->l_layfn->lf_LayRewrite)
#define LayResize		(*flayer->l_layfn->lf_LayResize)
#define LayRestore		(*flayer->l_layfn->lf_LayRestore)

#define LaySetCursor()	LGotoPos(flayer, flayer->l_x, flayer->l_y)
#define LayCanResize(l)	(l->l_layfn->LayResize != DefResize)

/* XXX: AArgh! think again! */

#define LAY_CALL_UP(fn) do				\
	{ 						\
	  struct layer *oldlay = flayer; 		\
	  struct canvas *oldcvlist, *cv;		\
	  debug("LayCallUp\n");				\
	  flayer = flayer->l_next;			\
	  oldcvlist = flayer->l_cvlist;			\
	  debug1("oldcvlist: %x\n", oldcvlist);		\
	  flayer->l_cvlist = oldlay->l_cvlist;		\
	  for (cv = flayer->l_cvlist; cv; cv = cv->c_lnext)	\
		cv->c_layer = flayer;			\
	  fn;						\
	  flayer = oldlay;				\
	  for (cv = flayer->l_cvlist; cv; cv = cv->c_lnext)	\
		cv->c_layer = flayer;			\
	  flayer->l_next->l_cvlist = oldcvlist;		\
	} while(0)

#define LAY_DISPLAYS(l, fn) do				\
	{ 						\
	  struct display *olddisplay = display;		\
	  struct canvas *cv;				\
	  for (display = displays; display; display = display->d_next) \
	    {						\
	      for (cv = D_cvlist; cv; cv = cv->c_next)	\
		if (cv->c_layer == l)			\
		  break;				\
	      if (cv == 0)				\
		continue;				\
	      fn;					\
	    }						\
	  display = olddisplay;				\
	} while(0)

