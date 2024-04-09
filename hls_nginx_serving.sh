#/bin/sh

# using podman or docker
podman run \
    -it \
    -p 5000:80 \
    --rm \
    -v ./hls_serving_file:/data/videos/ \
    -v ./nginx/nginx.conf:/etc/nginx/nginx.conf nginx
