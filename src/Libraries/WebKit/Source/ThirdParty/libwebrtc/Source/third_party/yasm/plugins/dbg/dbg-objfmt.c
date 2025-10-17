/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#include <stdio.h>
#include <libyasm.h>

#define N_(String) (String)

typedef struct yasm_objfmt_dbg {
    yasm_objfmt_base objfmt;        /* base structure */

    FILE *dbgfile;
} yasm_objfmt_dbg;

yasm_objfmt_module yasm_dbg_LTX_objfmt;


static yasm_objfmt *
dbg_objfmt_create(yasm_object *object)
{
    yasm_objfmt_dbg *objfmt_dbg = yasm_xmalloc(sizeof(yasm_objfmt_dbg));

    objfmt_dbg->objfmt.module = &yasm_dbg_LTX_objfmt;

    objfmt_dbg->dbgfile = tmpfile();
    if (!objfmt_dbg->dbgfile) {
        fprintf(stderr, N_("could not open temporary file"));
        return 0;
    }
    fprintf(objfmt_dbg->dbgfile, "PLUGIN create()\n");
    return (yasm_objfmt *)objfmt_dbg;
}

static void
dbg_objfmt_output(yasm_object *object, FILE *f, int all_syms,
                  yasm_errwarns *errwarns)
{
    yasm_objfmt_dbg *objfmt_dbg = (yasm_objfmt_dbg *)object->objfmt;
    char buf[1024];
    size_t i;

    /* Copy temp file to real output file */
    rewind(objfmt_dbg->dbgfile);
    while ((i = fread(buf, 1, 1024, objfmt_dbg->dbgfile))) {
        if (fwrite(buf, 1, i, f) != i)
            break;
    }

    /* Reassign objfmt debug file to output file */
    fclose(objfmt_dbg->dbgfile);
    objfmt_dbg->dbgfile = f;

    fprintf(objfmt_dbg->dbgfile, "PLUGIN output(f, object->\n");
    yasm_object_print(object, objfmt_dbg->dbgfile, 1);
    fprintf(objfmt_dbg->dbgfile, "%d)\n", all_syms);
    fprintf(objfmt_dbg->dbgfile, " Symbol Table:\n");
    yasm_symtab_print(object->symtab, objfmt_dbg->dbgfile, 1);
}

static void
dbg_objfmt_destroy(yasm_objfmt *objfmt)
{
    yasm_objfmt_dbg *objfmt_dbg = (yasm_objfmt_dbg *)objfmt;
    fprintf(objfmt_dbg->dbgfile, "PLUGIN destroy()\n");
    yasm_xfree(objfmt);
}

static yasm_section *
dbg_objfmt_add_default_section(yasm_object *object)
{
    yasm_objfmt_dbg *objfmt_dbg = (yasm_objfmt_dbg *)object->objfmt;
    yasm_section *retval;
    int isnew;

    retval = yasm_object_get_general(object, ".text", 0, 0, 0, &isnew, 0);
    if (isnew) {
        fprintf(objfmt_dbg->dbgfile, "(new) ");
        yasm_symtab_define_label(object->symtab, ".text",
            yasm_section_bcs_first(retval), 1, 0);
        yasm_section_set_default(retval, 1);
    }
    return retval;
}

static /*@observer@*/ /*@null@*/ yasm_section *
dbg_objfmt_section_switch(yasm_object *object, yasm_valparamhead *valparams,
                          /*@unused@*/ /*@null@*/
                          yasm_valparamhead *objext_valparams,
                          unsigned long line)
{
    yasm_objfmt_dbg *objfmt_dbg = (yasm_objfmt_dbg *)object->objfmt;
    yasm_valparam *vp;
    yasm_section *retval;
    int isnew;

    fprintf(objfmt_dbg->dbgfile, "PLUGIN section_switch(headp, ");
    yasm_vps_print(valparams, objfmt_dbg->dbgfile);
    fprintf(objfmt_dbg->dbgfile, ", ");
    yasm_vps_print(objext_valparams, objfmt_dbg->dbgfile);
    fprintf(objfmt_dbg->dbgfile, ", %lu), returning ", line);

    vp = yasm_vps_first(valparams);
    if (!yasm_vp_string(vp)) {
        fprintf(objfmt_dbg->dbgfile, "NULL\n");
        return NULL;
    }
    retval = yasm_object_get_general(object, yasm_vp_string(vp), 0, 0, 0,
                                     &isnew, line);
    if (isnew) {
        fprintf(objfmt_dbg->dbgfile, "(new) ");
        yasm_symtab_define_label(object->symtab, vp->val,
                                 yasm_section_bcs_first(retval), 1, line);
    }
    yasm_section_set_default(retval, 0);
    fprintf(objfmt_dbg->dbgfile, "\"%s\" section\n", vp->val);
    return retval;
}

static /*@observer@*/ /*@null@*/ yasm_symrec *
dbg_objfmt_get_special_sym(yasm_object *object, const char *name,
                           const char *parser)
{
    yasm_objfmt_dbg *objfmt_dbg = (yasm_objfmt_dbg *)object->objfmt;
    fprintf(objfmt_dbg->dbgfile,
            "PLUGIN get_special_sym(object, \"%s\", \"%s\")\n",
            name, parser);
    return NULL;
}

/* Define valid debug formats to use with this object format */
static const char *dbg_objfmt_dbgfmt_keywords[] = {
    "null",
    NULL
};

/* Define objfmt structure -- see objfmt.h for details */
yasm_objfmt_module yasm_dbg_LTX_objfmt = {
    "Trace of all info passed to object format module",
    "dbg",
    "dbg",
    32,
    dbg_objfmt_dbgfmt_keywords,
    "null",
    NULL,       /* no directives */
    NULL,       /* no standard macros */
    dbg_objfmt_create,
    dbg_objfmt_output,
    dbg_objfmt_destroy,
    dbg_objfmt_add_default_section,
    dbg_objfmt_section_switch,
    dbg_objfmt_get_special_sym
};
