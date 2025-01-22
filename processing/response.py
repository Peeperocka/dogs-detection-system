from utils.logger import logger


class Response:
    def handle_pack(self, clusters):
        if clusters:
            pack_sizes = [len(cluster) for cluster in clusters if cluster]
            if pack_sizes:
                logger.info(f"Обнаружены стаи собак, размеры стай: {pack_sizes}")
        else:
            logger.info("Собаки не были сгруппированы в стаи.")
