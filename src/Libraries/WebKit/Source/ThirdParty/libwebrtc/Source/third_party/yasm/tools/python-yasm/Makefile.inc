/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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

PYBINDING_DEPS  = tools/python-yasm/bytecode.pxi
PYBINDING_DEPS += tools/python-yasm/errwarn.pxi
PYBINDING_DEPS += tools/python-yasm/expr.pxi
PYBINDING_DEPS += tools/python-yasm/floatnum.pxi
PYBINDING_DEPS += tools/python-yasm/intnum.pxi
PYBINDING_DEPS += tools/python-yasm/symrec.pxi
PYBINDING_DEPS += tools/python-yasm/value.pxi

EXTRA_DIST += tools/python-yasm/pyxelator/cparse.py
EXTRA_DIST += tools/python-yasm/pyxelator/genpyx.py
EXTRA_DIST += tools/python-yasm/pyxelator/ir.py
EXTRA_DIST += tools/python-yasm/pyxelator/lexer.py
EXTRA_DIST += tools/python-yasm/pyxelator/node.py
EXTRA_DIST += tools/python-yasm/pyxelator/parse_core.py
EXTRA_DIST += tools/python-yasm/pyxelator/work_unit.py
EXTRA_DIST += tools/python-yasm/pyxelator/wrap_yasm.py
EXTRA_DIST += tools/python-yasm/setup.py
EXTRA_DIST += tools/python-yasm/yasm.pyx
EXTRA_DIST += $(PYBINDING_DEPS)

if HAVE_PYTHON_BINDINGS

# Use Pyxelator to generate Pyrex function headers.
_yasm.pxi: ${HEADERS}
	@rm -rf .tmp
	@mkdir .tmp
	$(PYTHON) $(srcdir)/tools/python-yasm/pyxelator/wrap_yasm.py \
		"YASM_DIR=${srcdir}" "CPP=${CPP}" "CPPFLAGS=${CPPFLAGS}"
	@rm -rf .tmp

CLEANFILES += _yasm.pxi

# Need to build a local copy of the main Pyrex input file to include _yasm.pxi
# from the build directory.  Also need to fixup the other .pxi include paths.
yasm.pyx: $(srcdir)/tools/python-yasm/yasm.pyx
	sed -e 's,^include "\([^_]\),include "${srcdir}/tools/python-yasm/\1,' \
		$(srcdir)/tools/python-yasm/yasm.pyx > $@

CLEANFILES += yasm.pyx

# Actually run Cython
yasm_python.c: yasm.pyx _yasm.pxi $(PYBINDING_DEPS)
	$(PYTHON) -c "from Cython.Compiler.Main import main; main(command_line=1)" \
		-o $@ yasm.pyx

CLEANFILES += yasm_python.c

# Now the Python build magic...
python-setup.txt: Makefile
	echo "includes=${DEFS} ${DEFAULT_INCLUDES} ${INCLUDES} ${AM_CPPFLAGS} ${CPPFLAGS}" > python-setup.txt
	echo "sources=${libyasm_a_SOURCES} ${nodist_libyasm_a_SOURCES}" >> python-setup.txt
	echo "srcdir=${srcdir}" >> python-setup.txt
	echo "gcc=${GCC}" >> python-setup.txt

CLEANFILES += python-setup.txt

.python-build: python-setup.txt yasm_python.c ${libyasm_a_SOURCES} ${nodist_libyasm_a_SOURCES}
	$(PYTHON) `test -f tools/python-yasm/setup.py || echo '$(srcdir)/'`tools/python-yasm/setup.py build
	touch .python-build
python-build: .python-build

CLEANFILES += .python-build

python-install: .python-build
	$(PYTHON) `test -f tools/python-yasm/setup.py || echo '$(srcdir)/'`tools/python-yasm/setup.py install "--install-lib=$(DESTDIR)$(pythondir)"

python-uninstall:
	rm -f `$(PYTHON) -c "import sys;sys.path.insert(0, '${DESTDIR}${pythondir}'); import yasm; print yasm.__file__"`

else

python-build:
python-install:
python-uninstall:

endif

EXTRA_DIST += tools/python-yasm/tests/Makefile.inc
include tools/python-yasm/tests/Makefile.inc
