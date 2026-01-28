# Podman Guidence
The purpose of this document is to define the steps of 
- building an image `podman build -t {tagname} -f {pathtodockerffile}`
- create a container for the image `podman run -d --name {containername} {imagename}:latest /bin/bash`
- copy files into the container `podman cp . {containername}:/src`
- attach to container ` podman exec -itw /src {containername} /bin/bash`
- copy file out of the container `podman cp {containername}:{sourcefilepath} {targetfilepath}`