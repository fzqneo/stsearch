#!/bin/sh

set -e
ARTIFACT="$1"
INSTALLER="${ARTIFACT}.install"

[ "${ARTIFACT}" ] || { echo "Usage: $0 <artifact.tgz>"; exit 1; }

echo ""
echo "Building installer with artifact ${ARTIFACT}"
echo "Will create ${INSTALLER}"
echo ""

cat - ${ARTIFACT} > ${INSTALLER} << EOF
#!/usr/bin/env bash

set -e
function cleanup {
    rm -rf \$TMPDIR;
}
trap cleanup EXIT

# Resolve installation destination
INSTDIR="\$HOME/.diamond"
read -e -p "Select installation directory (\$INSTDIR) [ENTER]:"
echo    # (optional) move to a new line
if [[ ! -z \$REPLY ]]; then
    INSTDIR=\$REPLY
fi

INSTDIR=\$(realpath \$INSTDIR)

if [[ ! -d "\$INSTDIR" ]]; then
    echo "Directory \$INSTDIR doesn't exist" 1>&2
    exit 1
fi

echo ""
echo "Diamond New Filters Self Extracting Installer"
echo "Installing filters to \$INSTDIR"
echo ""

# Decompress payload to a temp dir
TMPDIR=\$(mktemp -d /tmp/diamond_filter_selfextract.XXXXXX)
ARCHIVE=\$(awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0; }' \$0)
tail -n+\$ARCHIVE \$0 | tar xz -C \$TMPDIR

# Copy files to installation destination
(cd \$TMPDIR; cp -v -r diamond/* \$INSTDIR/ )

exit 0
__ARCHIVE_BELOW__
EOF

chmod +x ${INSTALLER}
echo "${INSTALLER} created."
