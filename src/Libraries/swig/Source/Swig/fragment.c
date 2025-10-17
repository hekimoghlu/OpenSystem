/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
char cvsroot_fragment_c[] = "$Id: fragment.c 9632 2007-01-03 20:58:19Z beazley $";

#include "swig.h"
#include "swigwarn.h"

static Hash *fragments = 0;
static Hash *looking_fragments = 0;
static int debug = 0;


/* -----------------------------------------------------------------------------
 * Swig_fragment_register()
 *
 * Add a fragment. Use the original Node*, so, if something needs to be
 * changed, lang.cxx doesn't nedd to be touched again.
 * ----------------------------------------------------------------------------- */

void Swig_fragment_register(Node *fragment) {
  if (Getattr(fragment, "emitonly")) {
    Swig_fragment_emit(fragment);
    return;
  } else {
    String *name = Copy(Getattr(fragment, "value"));
    String *type = Getattr(fragment, "type");
    if (type) {
      SwigType *rtype = SwigType_typedef_resolve_all(type);
      String *mangle = Swig_string_mangle(type);
      Append(name, mangle);
      Delete(mangle);
      Delete(rtype);
      if (debug)
	Printf(stdout, "register fragment %s %s\n", name, type);
    }
    if (!fragments) {
      fragments = NewHash();
    }
    if (!Getattr(fragments, name)) {
      String *section = Copy(Getattr(fragment, "section"));
      String *ccode = Copy(Getattr(fragment, "code"));
      Hash *kwargs = Getattr(fragment, "kwargs");
      Setmeta(ccode, "section", section);
      if (kwargs) {
	Setmeta(ccode, "kwargs", kwargs);
      }
      Setattr(fragments, name, ccode);
      if (debug)
	Printf(stdout, "registering fragment %s %s\n", name, section);
      Delete(section);
      Delete(ccode);
    }
    Delete(name);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_fragment_emit()
 *
 * Emit a fragment
 * ----------------------------------------------------------------------------- */

static
char *char_index(char *str, char c) {
  while (*str && (c != *str))
    ++str;
  return (c == *str) ? str : 0;
}

void Swig_fragment_emit(Node *n) {
  String *code;
  char *pc, *tok;
  String *t;
  String *mangle = 0;
  String *name = 0;
  String *type = 0;

  if (!fragments) {
    Swig_warning(WARN_FRAGMENT_NOT_FOUND, Getfile(n), Getline(n), "Fragment '%s' not found.\n", name);
    return;
  }


  name = Getattr(n, "value");
  if (!name) {
    name = n;
  }
  type = Getattr(n, "type");
  if (type) {
    mangle = Swig_string_mangle(type);
  }

  if (debug)
    Printf(stdout, "looking fragment %s %s\n", name, type);
  t = Copy(name);
  tok = Char(t);
  pc = char_index(tok, ',');
  if (pc)
    *pc = 0;
  while (tok) {
    String *name = NewString(tok);
    if (mangle)
      Append(name, mangle);
    if (looking_fragments && Getattr(looking_fragments, name)) {
      return;
    }
    code = Getattr(fragments, name);
    if (debug)
      Printf(stdout, "looking subfragment %s\n", name);
    if (code && (Strcmp(code, "ignore") != 0)) {
      String *section = Getmeta(code, "section");
      Hash *nn = Getmeta(code, "kwargs");
      if (!looking_fragments)
	looking_fragments = NewHash();
      Setattr(looking_fragments, name, "1");
      while (nn) {
	if (Equal(Getattr(nn, "name"), "fragment")) {
	  if (debug)
	    Printf(stdout, "emitting fragment %s %s\n", nn, type);
	  Setfile(nn, Getfile(n));
	  Setline(nn, Getline(n));
	  Swig_fragment_emit(nn);
	}
	nn = nextSibling(nn);
      }
      if (section) {
	File *f = Swig_filebyname(section);
	if (!f) {
	  Swig_error(Getfile(code), Getline(code), "Bad section '%s' for code fragment '%s'\n", section, name);
	} else {
	  if (debug)
	    Printf(stdout, "emitting subfragment %s %s\n", name, section);
	  if (debug)
	    Printf(f, "/* begin fragment %s */\n", name);
	  Printf(f, "%s\n", code);
	  if (debug)
	    Printf(f, "/* end fragment %s */\n\n", name);
	  Setattr(fragments, name, "ignore");
	  Delattr(looking_fragments, name);
	}
      }
    } else if (!code && type) {
      SwigType *rtype = SwigType_typedef_resolve_all(type);
      if (!Equal(type, rtype)) {
	String *name = Copy(Getattr(n, "value"));
	String *mangle = Swig_string_mangle(type);
	Append(name, mangle);
	Setfile(name, Getfile(n));
	Setline(name, Getline(n));
	Swig_fragment_emit(name);
	Delete(mangle);
	Delete(name);
      }
      Delete(rtype);
    }

    if (!code) {
      Swig_warning(WARN_FRAGMENT_NOT_FOUND, Getfile(n), Getline(n), "Fragment '%s' not found.\n", name);
    }
    tok = pc ? pc + 1 : 0;
    if (tok) {
      pc = char_index(tok, ',');
      if (pc)
	*pc = 0;
    }
    Delete(name);
  }
  Delete(t);
}
