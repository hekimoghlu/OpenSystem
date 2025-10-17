/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
char cvsroot_parms_c[] = "$Id: parms.c 11080 2009-01-24 13:15:51Z bhy $";

#include "swig.h"

/* ------------------------------------------------------------------------
 * NewParm()
 *
 * Create a new parameter from datatype 'type' and name 'name'.
 * ------------------------------------------------------------------------ */

Parm *NewParm(SwigType *type, const_String_or_char_ptr name) {
  Parm *p = NewHash();
  set_nodeType(p, "parm");
  if (type) {
    SwigType *ntype = Copy(type);
    Setattr(p, "type", ntype);
    Delete(ntype);
  }
  Setattr(p, "name", name);
  return p;
}

/* ------------------------------------------------------------------------
 * CopyParm()
 * ------------------------------------------------------------------------ */

Parm *CopyParm(Parm *p) {
  Parm *np = NewHash();
  Iterator ki;
  for (ki = First(p); ki.key; ki = Next(ki)) {
    if (DohIsString(ki.item)) {
      DOH *c = Copy(ki.item);
      Setattr(np,ki.key,c);
      Delete(c);
    }
  }
  Setfile(np, Getfile(p));
  Setline(np, Getline(p));
  return np;
}

/* ------------------------------------------------------------------
 * CopyParmListMax()
 * CopyParmList()
 * ------------------------------------------------------------------ */

ParmList *CopyParmListMax(ParmList *p, int count) {
  Parm *np;
  Parm *pp = 0;
  Parm *fp = 0;

  if (!p)
    return 0;

  while (p) {
    if (count == 0) break;
    np = CopyParm(p);
    if (pp) {
      set_nextSibling(pp, np);
      Delete(np);
    } else {
      fp = np;
    }
    pp = np;
    p = nextSibling(p);
    count--;
  }
  return fp;
}

ParmList *CopyParmList(ParmList *p) {
  return CopyParmListMax(p,-1);
}

/* -----------------------------------------------------------------------------
 * int ParmList_numrequired().  Return number of required arguments
 * ----------------------------------------------------------------------------- */

int ParmList_numrequired(ParmList *p) {
  int i = 0;
  while (p) {
    SwigType *t = Getattr(p, "type");
    String *value = Getattr(p, "value");
    if (value)
      return i;
    if (!(SwigType_type(t) == T_VOID))
      i++;
    else
      break;
    p = nextSibling(p);
  }
  return i;
}

/* -----------------------------------------------------------------------------
 * int ParmList_len()
 * ----------------------------------------------------------------------------- */

int ParmList_len(ParmList *p) {
  int i = 0;
  while (p) {
    i++;
    p = nextSibling(p);
  }
  return i;
}

/* ---------------------------------------------------------------------
 * ParmList_str()
 *
 * Generates a string of parameters
 * ---------------------------------------------------------------------- */

String *ParmList_str(ParmList *p) {
  String *out = NewStringEmpty();
  while (p) {
    String *pstr = SwigType_str(Getattr(p, "type"), Getattr(p, "name"));
    Append(out, pstr);
    p = nextSibling(p);
    if (p) {
      Append(out, ",");
    }
    Delete(pstr);
  }
  return out;
}

/* ---------------------------------------------------------------------
 * ParmList_str_defaultargs()
 *
 * Generates a string of parameters including default arguments
 * ---------------------------------------------------------------------- */

String *ParmList_str_defaultargs(ParmList *p) {
  String *out = NewStringEmpty();
  while (p) {
    String *value = Getattr(p, "value");
    String *pstr = SwigType_str(Getattr(p, "type"), Getattr(p, "name"));
    Append(out, pstr);
    if (value) {
      Printf(out, "=%s", value);
    }
    p = nextSibling(p);
    if (p) {
      Append(out, ",");
    }
    Delete(pstr);
  }
  return out;
}

/* ---------------------------------------------------------------------
 * ParmList_protostr()
 *
 * Generate a prototype string.
 * ---------------------------------------------------------------------- */

String *ParmList_protostr(ParmList *p) {
  String *out = NewStringEmpty();
  while (p) {
    String *pstr = SwigType_str(Getattr(p, "type"), 0);
    Append(out, pstr);
    p = nextSibling(p);
    if (p) {
      Append(out, ",");
    }
    Delete(pstr);
  }
  return out;
}

/* ---------------------------------------------------------------------
 * ParmList_has_defaultargs()
 *
 * Returns 1 if the parameter list passed in is has one or more default
 * arguments. Otherwise returns 0.
 * ---------------------------------------------------------------------- */

int ParmList_has_defaultargs(ParmList *p) {
  while (p) {
    if (Getattr(p, "value")) {
      return 1;
    }
    p = nextSibling(p);
  }
  return 0;
}
